python watertight.py --input ./mesh.glb --output ./watered_mesh.obj

python sharp.py --input ./watered_mesh.obj --output ./sharped_mesh.npz

torchrun sample_single.py --npz_path ./sharped_mesh.npz