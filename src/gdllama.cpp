#include "gdllama.h"

#include <algorithm>

#define ERROR(fmt, ...) fprintf(stderr, "ERROR: %s:%d:" fmt, __FILE__, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

#if 0
#define LOG(fmt, ...) fprintf(stderr, "LOG: " fmt __VA_OPT__(, ) __VA_ARGS__)
#else
#define LOG(fmt, ...) (void)0
#endif

GDLlama::GDLlama() {
}

GDLlama::~GDLlama() {
	deinit();
}

void GDLlama::thread_callback() {
  // Wait until model is loaded.
	std::string model_filename;
	{
		std::unique_lock<std::mutex> guard(mtx);
		if (!model_to_load.has_value()) {
			model_avail_cv.wait(guard, [this] { return model_to_load.has_value(); });
		}
		model_filename = *model_to_load;
	}
	thread_load_model(model_filename);
  LOG("> Model loaded.\n");

	while (1) {
		Request *req = nullptr;
		{
			std::unique_lock<std::mutex> guard(mtx);
			request_avail_cv.wait(guard, [this] { return !requests.empty() || should_quit; });
			if (should_quit) {
				LOG("> Got quit request.\n");
				break;
			}
			req = requests[0].get();
		}
    LOG("> Got request '%s'.\n", req->prompt.c_str());
    thread_submit(*req);
		{
			std::unique_lock<std::mutex> guard(mtx);
			requests.erase(requests.begin());
		}
	}
  LOG("> Thread finished.\n");
}

void GDLlama::thread_load_model(const std::string &filename) {
	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = 0; // TODO: make configurable?

	auto *new_model = llama_model_load_from_file(filename.c_str(), model_params);
	// TODO: better failure notifications
	if (!new_model) {
		ERROR("Failed to load model '%s'\n", filename.c_str());
		return;
	}
	const auto *new_vocab = llama_model_get_vocab(new_model);

	const int n_ctx = 32768;
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = n_ctx;
	ctx_params.n_batch = n_ctx;

	auto *new_ctx = llama_init_from_model(new_model, ctx_params);
	if (!new_ctx) {
		ERROR("Failed to create context.\n");
		return;
	}

	auto *new_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
	llama_sampler_chain_add(new_sampler, llama_sampler_init_temp(0.6f));
	llama_sampler_chain_add(new_sampler, llama_sampler_init_top_p(0.95f, 0));
	llama_sampler_chain_add(new_sampler, llama_sampler_init_top_k(20));
	llama_sampler_chain_add(new_sampler, llama_sampler_init_min_p(0.00f, 0));
	llama_sampler_chain_add(new_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
	llama_sampler_chain_add(
			new_sampler, llama_sampler_init_penalties(llama_vocab_n_tokens(new_vocab),
													  128, // last n tokens to penalize (0 = disable penalty)
													  1.0, // must be > 0.0, 1.0 = disabled
													  0.0, // must be finite, 0.0 = disabled
													  1.5)); // must be finite, 0.0 = disabled

	model = new_model;
	vocab = new_vocab;
	ctx = new_ctx;
	sampler = new_sampler;
}

void GDLlama::thread_submit(const Request &req) {
	std::string response;

	const bool is_first =
			llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;

	// tokenize the prompt
	const int n_prompt_tokens = -llama_tokenize(
			vocab, req.prompt.c_str(), req.prompt.size(), NULL, 0, is_first, true);
	std::vector<llama_token> prompt_tokens(n_prompt_tokens);
	if (llama_tokenize(vocab, req.prompt.c_str(), req.prompt.size(),
					   prompt_tokens.data(), prompt_tokens.size(), is_first,
					   true) < 0) {
		GGML_ABORT("failed to tokenize the prompt\n");
	}

	// prepare a batch for the prompt
	llama_batch batch =
			llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
	llama_token new_token_id;
	while (true) {
		// Check if this request was canceled.
		if (req.canceled) {
			response += "<<CANCELED>>\n";
			break;
		}

		// check if we have enough space in the context to evaluate this batch
		int n_ctx = llama_n_ctx(ctx);
		int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
		if (n_ctx_used + batch.n_tokens > n_ctx) {
			ERROR("context size exceeded\n");
			exit(0);
		}

		int ret = llama_decode(ctx, batch);
		if (ret != 0) {
			GGML_ABORT("failed to decode, ret = %d\n", ret);
		}

		auto token = llama_get_sampled_token_ith(ctx, -1);

		// sample the next token
		new_token_id = llama_sampler_sample(sampler, ctx, -1);

		// is it an end of generation?
		if (llama_vocab_is_eog(vocab, new_token_id)) {
			break;
		}

		// convert the token to a string, print it and add it to the response
		char buf[256];
		int n =
				llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
		if (n < 0) {
			GGML_ABORT("failed to convert token to piece\n");
		}
		std::string piece(buf, n);
#if 1
		printf("%s", piece.c_str());
		fflush(stdout);
#endif
		response += piece;

		// prepare the next batch with the sampled token
		batch = llama_batch_get_one(&new_token_id, 1);
	}

	{
		std::unique_lock guard(mtx);
		responses.push_back(Response{ req.id, response });
	}
}

void GDLlama::_bind_methods() {
	godot::ClassDB::bind_method(D_METHOD("init", "model_filename"), &GDLlama::init);
	godot::ClassDB::bind_method(D_METHOD("deinit"), &GDLlama::deinit);
	godot::ClassDB::bind_method(D_METHOD("submit", "id", "prompt"), &GDLlama::submit);
	godot::ClassDB::bind_method(D_METHOD("cancel", "id"), &GDLlama::cancel);
	godot::ClassDB::bind_method(D_METHOD("is_done", "id"), &GDLlama::is_done);
	godot::ClassDB::bind_method(D_METHOD("get_response", "id"), &GDLlama::get_response);
}

void GDLlama::init(const String &model_filename) {
	if (th) {
		ERROR("GDLlama already initialized.\n");
		return;
	}
	th = std::make_unique<std::jthread>(&GDLlama::thread_callback, this);

	LOG("Initializing GDLlama with model: '%s'...\n", model_filename.utf8().ptr());
	{
		std::unique_lock guard(mtx);
		if (model_to_load.has_value()) {
			ERROR("Model '%s' already loaded.\n", model_to_load->c_str());
			return;
		}
		model_to_load = std::string(model_filename.utf8().ptr());
	}
	model_avail_cv.notify_one();
	LOG("GDLlama initialized.\n");
}

void GDLlama::deinit() {
	LOG("Deinitializing GDLlama...\n");
  should_quit = true;
  request_avail_cv.notify_one();
  th->join();
  th.release();

  model_to_load.reset();
  requests.clear();
  responses.clear();

	llama_sampler_free(sampler);
	llama_free(ctx);
	llama_model_free(model);
	LOG("GDLlama deinitialized.\n");
}

void GDLlama::submit(int id, const String &prompt) {
	{
		std::unique_lock guard(mtx);
		requests.push_back(std::make_unique<Request>(id, prompt.utf8().ptr()));
	}
	request_avail_cv.notify_one();
}

void GDLlama::cancel(int id) {
	LOG("Canceling %d...\n", id);
	{
		std::unique_lock guard(mtx);
		auto iter = std::find_if(requests.begin(), requests.end(), [&](auto &req) {
			return req->id == id;
		});
		if (iter == requests.end()) {
			ERROR("Invalid id %d\n", id);
			return;
		}
		(*iter)->canceled = true;
    LOG("Canceled %d\n", id);
	}
}

bool GDLlama::is_done(int id) {
	{
		std::unique_lock guard(mtx);
		auto iter = std::find_if(requests.begin(), requests.end(), [&](auto &req) {
			return req->id == id;
		});
		return iter == requests.end();
	}
}

String GDLlama::get_response(int id) {
	std::unique_lock guard(mtx);
	auto iter = std::find_if(responses.begin(), responses.end(), [&](auto &res) {
		return res.id == id;
	});
	if (iter == responses.end())
		return "";
	std::string response = iter->response;
	responses.erase(iter);
	return String(response.c_str());
}

