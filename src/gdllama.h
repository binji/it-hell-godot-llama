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

class GDLlama : public RefCounted {
	GDCLASS(GDLlama, RefCounted)

  void thread_callback();
  void thread_load_model(const std::string &filename);
  void thread_submit(const Request &);

protected:
	static void _bind_methods();

public:
	GDLlama();
	~GDLlama() override;

	void init(const String &model_filename);
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
  std::optional<std::string> model_to_load;
  std::condition_variable model_avail_cv;
  std::vector<std::unique_ptr<Request>> requests;
  std::vector<Response> responses;
  std::condition_variable request_avail_cv;
};
