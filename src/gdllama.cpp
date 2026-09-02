#include "gdllama.h"

#include <algorithm>

#define ERROR(fmt, ...) fprintf(stderr, "ERROR: %s:%d:" fmt, __FILE__, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

#if 1
#define LOG(fmt, ...) fprintf(stderr, "LOG: " fmt __VA_OPT__(, ) __VA_ARGS__)
#else
#define LOG(fmt, ...) (void)0
#endif

void LlamaInitParams::_bind_methods() {
	godot::ClassDB::bind_method(D_METHOD("set_model_filename", "model_filename"), &LlamaInitParams::set_model_filename);
	godot::ClassDB::bind_method(D_METHOD("set_gpu_layers", "gpu_layers"), &LlamaInitParams::set_gpu_layers);
	godot::ClassDB::bind_method(D_METHOD("set_n_ctx", "n_ctx"), &LlamaInitParams::set_n_ctx);
	godot::ClassDB::bind_method(D_METHOD("set_temp", "temp"), &LlamaInitParams::set_temp);
	godot::ClassDB::bind_method(D_METHOD("set_top_p", "top_p"), &LlamaInitParams::set_top_p);
	godot::ClassDB::bind_method(D_METHOD("set_top_k", "top_k"), &LlamaInitParams::set_top_k);
	godot::ClassDB::bind_method(D_METHOD("set_min_p", "min_p"), &LlamaInitParams::set_min_p);
	godot::ClassDB::bind_method(D_METHOD("set_penalty_tokens", "penalty_tokens"), &LlamaInitParams::set_penalty_tokens);
	godot::ClassDB::bind_method(D_METHOD("set_presence_penalty", "presence_penalty"), &LlamaInitParams::set_presence_penalty);
}

void LlamaInitParams::set_model_filename(const String &model_filename) {
	data.model_filename = model_filename.utf8().ptr();
}

void LlamaInitParams::set_gpu_layers(int gpu_layers) {
	data.gpu_layers = gpu_layers;
}

void LlamaInitParams::set_n_ctx(int n_ctx) {
	data.n_ctx = n_ctx;
}

void LlamaInitParams::set_temp(float temp) {
	data.temp = temp;
}

void LlamaInitParams::set_top_p(float top_p) {
	data.top_p = top_p;
}

void LlamaInitParams::set_top_k(int top_k) {
	data.top_k = top_k;
}

void LlamaInitParams::set_min_p(float min_p) {
	data.min_p = min_p;
}

void LlamaInitParams::set_penalty_tokens(int penalty_tokens) {
	data.penalty_tokens = penalty_tokens;
}

void LlamaInitParams::set_presence_penalty(float presence_penalty) {
	data.presence_penalty = presence_penalty;
}

Llama::Llama() {
}

Llama::~Llama() {
	deinit();
}

void Llama::thread_callback() {
	// Wait until init params are given.
	InitParamsData init_params_data;
	{
		std::unique_lock<std::mutex> guard(mtx);
		if (!opt_init_params_data.has_value()) {
			model_avail_cv.wait(guard, [this] { return opt_init_params_data.has_value(); });
		}
		init_params_data = *opt_init_params_data;
	}
	thread_init(init_params_data);
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

void Llama::thread_init(const InitParamsData &data) {
	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = data.gpu_layers;

	auto *new_model = llama_model_load_from_file(data.model_filename.c_str(), model_params);
	// TODO: better failure notifications
	if (!new_model) {
		ERROR("Failed to load model '%s'\n", data.model_filename.c_str());
		return;
	}
	const auto *new_vocab = llama_model_get_vocab(new_model);

	const int n_ctx = data.n_ctx;
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = n_ctx;
	ctx_params.n_batch = n_ctx;

	auto *new_ctx = llama_init_from_model(new_model, ctx_params);
	if (!new_ctx) {
		ERROR("Failed to create context.\n");
		return;
	}

	auto *new_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
	llama_sampler_chain_add(new_sampler, llama_sampler_init_temp(data.temp));
	llama_sampler_chain_add(new_sampler, llama_sampler_init_top_p(data.top_p, 0));
	llama_sampler_chain_add(new_sampler, llama_sampler_init_top_k(data.top_k));
	llama_sampler_chain_add(new_sampler, llama_sampler_init_min_p(data.min_p, 0));
	llama_sampler_chain_add(new_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
	llama_sampler_chain_add(
			new_sampler,
			llama_sampler_init_penalties(
					llama_vocab_n_tokens(new_vocab),
					data.penalty_tokens, // last n tokens to penalize (0 = disable penalty)
					1.0, // must be > 0.0, 1.0 = disabled
					0.0, // must be finite, 0.0 = disabled
					data.presence_penalty)); // must be finite, 0.0 = disabled

	model = new_model;
	vocab = new_vocab;
	ctx = new_ctx;
	sampler = new_sampler;
}

void Llama::thread_submit(const Request &req) {
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

void Llama::_bind_methods() {
	godot::ClassDB::bind_method(D_METHOD("init", "init_params"), &Llama::init);
	godot::ClassDB::bind_method(D_METHOD("deinit"), &Llama::deinit);
	godot::ClassDB::bind_method(D_METHOD("submit", "id", "prompt"), &Llama::submit);
	godot::ClassDB::bind_method(D_METHOD("cancel", "id"), &Llama::cancel);
	godot::ClassDB::bind_method(D_METHOD("is_done", "id"), &Llama::is_done);
	godot::ClassDB::bind_method(D_METHOD("get_response", "id"), &Llama::get_response);
}

void Llama::init(const Ref<LlamaInitParams> &init_params) {
	if (th) {
		ERROR("Llama already initialized.\n");
		return;
	}
	th = std::make_unique<std::jthread>(&Llama::thread_callback, this);

	LOG("Initializing Llama with model: '%s'...\n", init_params->data.model_filename.c_str());
	{
		std::unique_lock guard(mtx);
		if (opt_init_params_data.has_value()) {
			ERROR("Llama already initialized.\n");
			return;
		}
		opt_init_params_data = init_params->data;
	}
	model_avail_cv.notify_one();
	LOG("Llama initialized.\n");
}

void Llama::deinit() {
  if (!th) {
    LOG("Llama already deinitialized.\n");
    return;
  }
	LOG("Deinitializing Llama...\n");
  should_quit = true;
  request_avail_cv.notify_one();
  th->join();
  th.release();

  opt_init_params_data.reset();
  requests.clear();
  responses.clear();

	llama_sampler_free(sampler);
	llama_free(ctx);
	llama_model_free(model);
	LOG("Llama deinitialized.\n");
}

void Llama::submit(int id, const String &prompt) {
	{
		std::unique_lock guard(mtx);
		requests.push_back(std::make_unique<Request>(id, prompt.utf8().ptr()));
	}
	request_avail_cv.notify_one();
}

void Llama::cancel(int id) {
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

bool Llama::is_done(int id) {
	{
		std::unique_lock guard(mtx);
		auto iter = std::find_if(requests.begin(), requests.end(), [&](auto &req) {
			return req->id == id;
		});
		return iter == requests.end();
	}
}

String Llama::get_response(int id) {
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

