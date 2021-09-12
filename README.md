# Setup

## Windows

1. Install CUDA toolkit from [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64).
   Note: lib has been tested with CUDA 11.04
2. Go to 'C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.4'
3. Copy the 'common' folder into this repo

### Clion

#### Cmake

4. open the project in Clion as 'Cmake project'
5. compile the programm. Note that execution will fail.
6. from the 'C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.4\bin\win64\release\' folder copy the 'freeglut.dll'
   and 'glew64.dll' into the './cmake-build-debug' folder created by clion
7. Run the programm again and now it should execute

### Visual Studio 2019

#### New Visual Studio Project (optional)

Only needed if you do not want to use the included .vcxproj file.

4. create a new `CUDA` project. Here v11.4
5. From 'C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.4' copy the 'bin' folder into this repo
6. Open the properties of the Solution
7. 'Configuration Properties >> C/C++ >> General' in 'AdditionalIncludeDirectories' add ';common/inc'
8. 'Configuration Properties >> Linker >> General' in 'Additional Library Directories' add ';common/lib/$(PlatformName)'
9. 'Configuration Properties >> Linker >> Input' in 'Additional Library Directories' add 'glew64.lib'
10. 'Configuration Properties >> General' change 'Output Directory' to '$(ProjectDir)bin/win64/$(Configuration)/'
5. For Visual Mode go
   to: ``Project > Properties > Configuration Properties > C/C++ > Preprocessor > Preprocessor Definitions`` and
   add `SIMVIZ`

#### Existing Visual Studio Project

4. Just open the included CUDA-OpenGL-Basic-Interaction.vcxproj file
5. For Visual Mode go
   to: ``Project > Properties > Configuration Properties > C/C++ > Preprocessor > Preprocessor Definitions`` and
   add `SIMVIZ`

## Linux

1. `sudo apt install nvidia-cuda-toolkit`
2. `sudo apt-get install freeglut3-dev`
3. `sudo apt-get install libglew-dev`
4. `sudo apt-get install libglfw3 libglfw3-dev` (???)

Use CMake

1. `mkdir build` to create a build folder
2. `cmake ..` or use `cmake -DHEADLESS=FALSE -DDEBUG=FALSE ..` see `Options`

### Installing latest CUDA version (optional)

see the [cuda instalation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

# Options

Change the resolution in `interactions.h`.

## Headless or Visual Mode

Headless mode will not visualize the data and therefore does not need the corresponding libraries. Can be used for
performance measurement without OpenGL or for using the data otherwise.

Activate by defining `HEADLESS`.

For cmake compile with `cmake -DHEADLESS=TRUE ..`

For Visual Studio go
to: ``Project > Properties > Configuration Properties > C/C++ > Preprocessor > Preprocessor Definitions`` and
remove `SIMVIZ`

## Debug Mode

Activate by defining `DEBUG`.

For cmake compile with `cmake -DEBUG=TRUE ..`

For Visual Studio go to: ``Project > Properties > Configuration Properties > Debugging > Environment`` and add `DEBUG`

# Sources

Large parts of Code taken from https://www.informit.com/articles/article.aspx?p=2455391&seqNum=2


