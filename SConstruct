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
    "CMAKE_BUILD_TYPE=Release",
    "LLAMA_BUILD_APP=OFF",
    "LLAMA_BUILD_EXAMPLES=OFF",
    "LLAMA_BUILD_SERVER=OFF",
    "LLAMA_BUILD_TESTS=OFF",
    "LLAMA_BUILD_TOOLS=OFF",
    "LLAMA_BUILD_UI=OFF",
    ])
llama_cpp = env.Command(
    target=f"cmake-install/bin/{LLAMA_SHLIB}",
    source="llama.cpp/CMakeLists.txt",
    action=[
        f"cmake -S llama.cpp -B cmake-build {llama_cpp_flags}",
        "cmake --build cmake-build",
        "cmake --build cmake-build -t install"
    ])

env.Append(CPPPATH=["cmake-install/include"])
env.Append(LIBPATH=["cmake-install/lib"])

# Install needed llama.cpp libraries to bin
DEPS_NAMES =    ["llama", "llama-common", "ggml-base", "ggml-cpu", "ggml"]
DEPS_VERSIONS = ["0.3.0", "0.3.0",        "0.22.0",    "0.22.0",   "0.22.0"]
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

lib_filename = shlib_name(libname, suffix)

env.Append(LIBS=DEPS_NAMES)
env.Append(LIBPATH=["cmake-install/lib"])
library = env.SharedLibrary(
    "bin/{}/{}".format(env['platform'], lib_filename),
    source=sources,
)
env.Depends(sources, llama_cpp)  # Make sure to install llama.cpp before compiling extension

defaults = []
for i, path in enumerate(DEPS_PATHS):
    env.Depends(path, llama_cpp)
    full_version = DEPS_VERSIONS[i]
    major_version = full_version[0]
    for version in ["", "." + major_version, "." + full_version]:
        path_version = path + version
        # Install deps to bin
        defaults.append(env.Install("bin/{}/".format(env["platform"]), path_version))
        # Install deps to projectdir
        defaults.append(env.Install("{}/bin/{}/".format(projectdir, env["platform"]), path_version))

defaults.append(env.Install("{}/bin/{}/".format(projectdir, env["platform"]), library))

default_args = [library, llama_cpp] + defaults
Default(*default_args)
