
# Ensure Jax matches the CUDA that will be used to build Madrona MJX
!pip uninstall -y jax
!pip install jax["cuda12_local"]==0.4.35

! sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev mesa-common-dev

# Prevent Python from trying to import the source packages.
mkdir modules
cd modules

# -- 7) Install madrona-mjx
echo "Cloning madrona-mjx..."
git clone https://github.com/shacklettbp/madrona_mjx.git

cd madrona_mjx
git submodule update --init --recursive

mkdir build
cd build
cmake -DLOAD_VULKAN=OFF ..
make -j 8

cd ..
pip install -e .
