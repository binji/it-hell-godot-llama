extends SceneTree

func getsecs() -> float:
	return Time.get_unix_time_from_system()

func _init() -> void:
	var init := LlamaInitParams.new()
	init.set_model_filename("./Qwen3-4B-Q6_K.gguf")
	var llama := Llama.new()
	llama.init(init)
	llama.submit(0, "<im_start>user\nhello, briefly describe what mohair is<im_end>\n<im_start>assistant\n<think>\n\n</think>\n\n")
	# llama.submit(0, "<im_start>user\nhello<im_end>\n<im_start>assistant\n<think>\n\n</think>\n\n")
	var start = getsecs()
	print("starting at: ", start)
	while !llama.is_done(0):
		var now = getsecs()
		if false: # now - start > 10.0:
			print("timeout at ", now - start, "!\n")
			llama.cancel(0)
			break
		pass
	print("finished at: ", getsecs() - start)
	print("got response: '", llama.get_response(0), "'\n")
	print("done.")
	llama.deinit()
	quit()
