#!/usr/bin/env python
import os
import sys

from methods import print_error

libname = "it_hell_llama"
projectdir = "project"

localEnv = Environment(
    tools=["default"],
    PLATFORM="",
    ENV={
        'HOME': os.environ.get('HOME'),
    })

# Build profiles can be used to decrease compile times.
# You can either specify "disabled_classes", OR
# explicitly specify "enabled_classes" which disables all other classes.
# Modify the example file as needed and uncomment the line below or
# manually specify the build_profile parameter when running SCons.

# localEnv["build_profile"] = "build_profile.json"

customs = ["custom.py"]
customs = [os.path.abspath(path) for path in customs]

opts = Variables(customs, ARGUMENTS)
opts.Update(localEnv)

Help(opts.GenerateHelpText(localEnv))

env = localEnv.Clone()

def shlib_name(baselib, suffix=""):
    return "{}{}{}{}".format(env.subst('$SHLIBPREFIX'), baselib, suffix, env.subst('$SHLIBSUFFIX'))

# llama.cpp
LLAMA_SHLIB = shlib_name("llama")

llama_cpp_flags = ' '.join("-D"+flag for flag in [
    f"CMAKE_INSTALL_PREFIX={os.path.abspath('.')}/cmake-install",
    f"LLAMA_BUILD_APP=OFF",
    f"LLAMA_BUILD_EXAMPLES=OFF",
    f"LLAMA_BUILD_SERVER=OFF",
    f"LLAMA_BUILD_TESTS=OFF",
    f"LLAMA_BUILD_TOOLS=OFF",
    f"LLAMA_BUILD_UI=OFF",
    ])
llama_cpp_configure = env.Command(
    source="llama.cpp/CMakeLists.txt",
    target="cmake-build/CMakeCache.txt",
    action=f"cmake -S llama.cpp -B cmake-build {llama_cpp_flags}")

llama_cpp_build = env.Command(
    target=f"cmake-build/lib/{LLAMA_SHLIB}",
    source="cmake-build/CMakeCache.txt",
    action="cmake --build cmake-build")
env.Depends(llama_cpp_build, llama_cpp_configure)

llama_cpp_install = env.Command(
    target=f"cmake-install/bin/{LLAMA_SHLIB}",
    source=f"cmake-build/bin/{LLAMA_SHLIB}",
    action="cmake --build cmake-build -t install")
env.Depends(llama_cpp_install, llama_cpp_build)

env.Append(CPPPATH=["cmake-install/include"])
env.Append(LIBPATH=["cmake-install/lib"])

# Install needed llama.cpp libraries to bin
DEPS_NAMES = ["llama", "llama-common", "ggml-base", "ggml-cpu", "ggml"]
DEPS_PATHS = [os.path.join("cmake-install/lib", shlib_name(lib)) for lib in DEPS_NAMES]

# godot-cpp
if not (os.path.isdir("godot-cpp") and os.listdir("godot-cpp")):
    print_error("""godot-cpp is not available within this folder, as Git submodules haven't been initialized.
Run the following command to download godot-cpp:

    git submodule update --init --recursive""")
    sys.exit(1)

env = SConscript("godot-cpp/SConstruct", {"env": env, "customs": customs})

env.Append(CXXFLAGS=["-std=c++20"])
env.Append(CPPPATH=["src/"])
sources = Glob("src/*.cpp")

if env["target"] in ["editor", "template_debug"]:
    try:
        doc_data = env.GodotCPPDocData("src/gen/doc_data.gen.cpp", source=Glob("doc_classes/*.xml"))
        sources.append(doc_data)
    except AttributeError:
        print("Not including class reference as we're targeting a pre-4.3 baseline.")

# .dev doesn't inhibit compatibility, so we don't need to key it.
# .universal just means "compatible with all relevant arches" so we don't need to key it.
suffix = env['suffix'].replace(".dev", "").replace(".universal", "")

lib_filename = shlib_name(libname)

env.Append(LIBS=DEPS_NAMES)
env.Append(LIBPATH=["cmake-install/lib"])
library = env.SharedLibrary(
    "bin/{}/{}".format(env['platform'], lib_filename),
    source=sources,
)
env.Depends(library, llama_cpp_install)

# Install deps to bin
deps_copy = env.Install("bin/{}/".format(env["platform"]), DEPS_PATHS)

# Install library and deps to projectdir
projdir_copy = env.Install("{}/bin/{}/".format(projectdir, env["platform"]),
                   [library] + DEPS_PATHS)

default_args = [library, deps_copy, projdir_copy, llama_cpp_install]
Default(*default_args)
