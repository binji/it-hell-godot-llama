#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <variant>

#include "godot_cpp/classes/ref_counted.hpp"
#include "godot_cpp/classes/wrapped.hpp"
#include "godot_cpp/variant/string.hpp"

#include "llama.h"

using namespace godot;

struct Request {
  int id;
  std::string prompt;
  std::atomic<bool> canceled = false;
};

struct Response {
	int id;
	std::string response;
};

struct InitParamsData {
	std::string model_filename;
	int gpu_layers = 0;
	int n_ctx = 32768;
	float temp = 0.7;
	float top_p = 0.85;
	int top_k = 20;
	float min_p = 0.0;
	int penalty_tokens = 128;
	float presence_penalty = 1.5;
};

struct LlamaInitParams : RefCounted {
	GDCLASS(LlamaInitParams, RefCounted)

protected:
	static void _bind_methods();

public:
	LlamaInitParams() = default;
	~LlamaInitParams() override = default;

	void set_model_filename(const String &model_filename);
	void set_gpu_layers(int gpu_layers);
	void set_n_ctx(int n_ctx);
	void set_temp(float temp);
	void set_top_p(float top_p);
	void set_top_k(int top_k);
	void set_min_p(float min_p);
	void set_penalty_tokens(int penalty_tokens);
	void set_presence_penalty(float presence_penalty);

  InitParamsData data;
};

class Llama : public RefCounted {
	GDCLASS(Llama, RefCounted)

  void thread_callback();
  void thread_init(const InitParamsData&);
  void thread_submit(const Request &);

protected:
	static void _bind_methods();

public:
	Llama();
	~Llama() override;

	void init(const Ref<LlamaInitParams> &init_params);
	void deinit();
	void submit(int id, const String &prompt);
	void cancel(int id);
	bool is_done(int id);
	String get_response(int id);

	llama_model *model;
	const llama_vocab *vocab;
	llama_context *ctx;
	llama_sampler *sampler;

  std::unique_ptr<std::jthread> th;
  std::atomic<bool> should_quit;
  std::mutex mtx;
  std::optional<InitParamsData> opt_init_params_data;
  std::condition_variable model_avail_cv;
  std::vector<std::unique_ptr<Request>> requests;
  std::vector<Response> responses;
  std::condition_variable request_avail_cv;
};
