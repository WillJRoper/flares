export pin_run_as_build="OrderedDict([('python', OrderedDict([('min_pin', 'x.x'), ('max_pin', 'x.x')])), ('r-base', OrderedDict([('min_pin', 'x.x.x'), ('max_pin', 'x.x.x')]))])"
export PYTHONNOUSERSITE="1"
export CONDA_NPY="114"
export cpu_optimization_target="nocona"
export PKG_CONFIG_PATH="/Users/willroper/anaconda3/envs/webbpsf-env/lib/pkgconfig"
export SRC_DIR="/Users/iraf/build/pembry/workspace/AstroConda/public_OSX-10.11_py3.5_np1.14/_dispatch/miniconda/conda-bld/webbpsf-data_1544778528500/work"
export ROOT="/Users/iraf/build/pembry/workspace/AstroConda/public_OSX-10.11_py3.5_np1.14/_dispatch/miniconda"
export LUA_VER="5"
export CONDA_BUILD="1"
export SYS_PREFIX="/Users/iraf/build/pembry/workspace/AstroConda/public_OSX-10.11_py3.5_np1.14/_dispatch/miniconda"
export CONDA_DEFAULT_ENV="/Users/willroper/anaconda3/envs/webbpsf-env"
export PY3K="1"
export fortran_compiler="gfortran"
export BUILD_PREFIX="/Users/willroper/anaconda3/envs/webbpsf-env"
export CONDA_BUILD_STATE="BUILD"
export PERL_VER="5.26"
export PKG_VERSION="0.8.0"
export r_base="3.4"
export CONDA_PY="35"
export CMAKE_GENERATOR="Unix Makefiles"
export SHLIB_EXT=".dylib"
export PKG_NAME="webbpsf-data"
export CONDA_PERL="5.26"
export OSX_ARCH="x86_64"
export RECIPE_DIR="/Users/iraf/build/pembry/workspace/astroconda/public_OSX-10.11_py3.5_np1.14/_dispatch/conda-recipes/webbpsf-data"
export PREFIX="/Users/willroper/anaconda3/envs/webbpsf-env"
export SYS_PYTHON="/Users/iraf/build/pembry/workspace/AstroConda/public_OSX-10.11_py3.5_np1.14/_dispatch/miniconda/bin/python"
export MACOSX_DEPLOYMENT_TARGET="10.9"
export LANG="en_US.UTF-8"
export CONDA_LUA="5"
export cxx_compiler="clangxx"
export DIRTY="1"
export PY_VER="3.5"
export c_compiler="clang"
export PATH="/Users/willroper/anaconda3/envs/webbpsf-env:/Users/willroper/anaconda3/envs/webbpsf-env/bin:/Users/iraf/build/pembry/workspace/AstroConda/public_OSX-10.11_py3.5_np1.14/_dispatch/miniconda/bin/:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"
export ignore_build_only_deps="python"
export pin_build_as_run="python"
export NPY_VER="1.14"
export NPY_DISTUTILS_APPEND_FLAGS="1"
export PKG_BUILDNUM="0"
export PIP_IGNORE_INSTALLED="True"
export R_VER="3.4"
export ARCH="64"
export HOME="/Users/iraf/build/pembry/workspace/AstroConda/public_OSX-10.11_py3.5_np1.14/webbpsf/home"
export target_platform="osx-64"
export PKG_HASH="1234567"
export CONDA_R="3.4"
export SUBDIR="osx-64"
export PKG_BUILD_STRING="placeholder"
export BUILD="x86_64-apple-darwin13.4.0"
export SP_DIR="/Users/willroper/anaconda3/envs/webbpsf-env/lib/python3.5/site-packages"
export CPU_COUNT="8"
export STDLIB_DIR="/Users/willroper/anaconda3/envs/webbpsf-env/lib/python3.5"
export cran_mirror="https://cran.r-project.org"
source "/Users/iraf/build/pembry/workspace/AstroConda/public_OSX-10.11_py3.5_np1.14/_dispatch/miniconda/bin/activate" "/Users/willroper/anaconda3/envs/webbpsf-env"
mkdir -p $PREFIX/share
mkdir -p $PREFIX/etc/conda/activate.d
mkdir -p $PREFIX/etc/conda/deactivate.d

# Different conda-build releases do different things
# SRC_DIR is "usedir" for all intents and purposes
usedir="$(pwd)"
if [[ -d ${usedir}/work ]]; then
    usedir=${usedir}/work
fi

rsync -a ${usedir}/* $PREFIX/share/webbpsf-data

echo "
export WEBBPSF_PATH=$PREFIX/share/webbpsf-data
" > $PREFIX/etc/conda/activate.d/webbpsf-data.sh

echo "
unset WEBBPSF_PATH
" > $PREFIX/etc/conda/deactivate.d/webbpsf-data.sh