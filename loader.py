import tqdm
import trimesh
import pyrender
import numpy as np
from pathlib import Path

from monai.data import DataLoader
from monai.data.dataset import Dataset as _MonaiDataset

from monai.transforms import Compose
from MatrixConvert import MatrixConverter

def get_matrix(PPA, PSA, SID):
	extrinsics = MatrixConverter.DICOMtoExtrinsics(
				positioner_primary_angle=PPA,
				positioner_secondary_angle=PSA,
				source_to_image_distance=SID
			)

	return MatrixConverter.extrinsicstoPose(extrinsics["R"], extrinsics["t"])

class MeshDataset(_MonaiDataset):
	"""The MeshDataset class is used to load stl files into memory.
	When used in conjunction with the RenderImage transform, it can be used for on-the-fly image generation from a mesh dataset.
	Each mesh is wrapped as a PyRender mesh object.
	"""

	def __init__(self, mesh_paths, image_transform=[], use_threadsafe=False):
		"""
		Args:
			mesh_paths (dict): A dictionary where the key is an index and the value is a string containing a path to a .stl file
		"""        
		self.meshes = {}

		_RenderImage = ThreadsafeRenderImage if use_threadsafe else RenderImage

		self.transform = Compose([
			_RenderImage(),
			*image_transform
		])

		self._load_meshes_to_cache(mesh_paths)

	def __getitem__(self, idx):
		"""
		Returns:
			pyrender.Mesh: The mesh object at index `idx`
		"""        
		assert idx in self.meshes, f"Mesh at index {idx} not found"

		sample = self.meshes.get(idx)
		sample = self.transform(sample)

		return sample

	def __len__(self):
		return len(self.meshes)

	def _load_meshes_to_cache(self, mesh_paths):
		for idx, path in tqdm.tqdm(mesh_paths.items()):
			mesh = trimesh.load(path)
			self.meshes[idx] = pyrender.Mesh.from_trimesh(mesh)


class RenderImage(object):
	"""
	The RenderImage transform renders an image or multiple images of the passed mesh using the PyRender library.
	"""     

	def __init__(self, image_size=[250,250], pixel_size=[.25,.25], focal_length=[4100,4100], sequence_type="list",
				ppa_list=[0,20], psa_list=[0,0], ppa_range=[-20,20], psa_range=[0,0], seed=0, n_images=2):  
	    """
        Args:
            image_size (list, optional): m x n pixels of the output images as [width, height]. Defaults to [250,250].
            pixel_size (list, optional): Physical size of the pixels in mm. Defaults to [.25,.25].
            focal_length (list, optional): Focal length of the camera in mm. Defaults to [4100,4100].
            sequence_type (str, optional): "list" if supplying a set of <PPA&PSA> or "random" to generate a set. Defaults to "list".
            ppa_list (list, optional): If "sequence_type" is "list", a list of PPAs to generate projections - zipped with psa_list. Defaults to [0,20].
            psa_list (list, optional): If "sequence_type" is "list", a list of PSAs to generate projections - zipped with ppa_list. Defaults to [0,0].
            ppa_range (list, optional): If "sequence_type" is "random", the range of PPAs to generate projections from. Defaults to [-20,20].
            psa_range (list, optional): If "sequence_type" is "random", the range of PSAs to generate projections from. Defaults to [0,0].
            seed (int, optional): If "sequence_type" is "random", the random seed for the generator. 0 seeds from system clock. Defaults to 0.
            n_images (int, optional): If "sequence_type" is "random", the number of images to generate. Defaults to 2.
        """         
		self._scene = None
		self._renderer = None
		self._light_obj = None
		self._cam_obj = None

		self.image_size = image_size
		self.pixel_size = pixel_size
		self.focal_length = focal_length

		self.sequence_type = sequence_type # either "list" or "random"
		self.ppa_list = ppa_list
		self.psa_list = psa_list
		self.ppa_range = ppa_range
		self.psa_range = psa_range
		self.seed = seed
		self.n_images = n_images

		if sequence_type == "list":
			assert len(ppa_list) == len(psa_list), f"'ppa_list' dimension must match 'psa_list': {len(ppa_list)} vs {len(psa_list)}"
			self.n_images = len(ppa_list)
		elif sequence_type == "random":
			#TODO: implement random generation using seed
			pass

		self._instantiate_pyrender_scene()

	def _instantiate_pyrender_scene(self):
		self._scene = pyrender.Scene()
		self._renderer = pyrender.OffscreenRenderer(self.image_size[0],self.image_size[1])

		light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
		cam = pyrender.IntrinsicsCamera(
				fx=self.focal_length[0],
				fy=self.focal_length[0],
				cx=self.image_size[0]/2,
				cy=self.image_size[1]/2,
				zfar=100000000000000 # `Infinite` clipping
			)

		self._light_obj = self._scene.add(light)
		self._cam_obj = self._scene.add(cam)

	def __call__(self, mesh):
		mesh_obj = self._scene.add(mesh)

		imgs = []
		for i in range(self.n_images):
			pose = get_matrix(self.ppa_list[i], self.psa_list[i], self.pixel_size[0]*self.focal_length[0])
			imgs.append(
				self._get_one_image(pose)
			)

		self._scene.remove_node(mesh_obj)

		return np.dstack(imgs)

	def _get_one_image(self, pose):
		self._scene.set_pose(self._light_obj, pose)
		self._scene.set_pose(self._cam_obj, pose)

		image_rgb, depth = self._renderer.render(self._scene)

		# Convert to single channel (greyscale)
		image = np.array(image_rgb[:, :, 0])*.299 + np.array(image_rgb[:, :, 1])*.587 + np.array(image_rgb[:, :, 2])*.114
		return image.astype(np.uint8)  


class ThreadsafeRenderImage(object): 

	def __init__(self, image_size=[250,250], pixel_size=[.25,.25], focal_length=[4100,4100], sequence_type="list",
				ppa_list=[0,20], psa_list=[0,0], ppa_range=[-20,20], psa_range=[0,0], seed=0, n_images=2):
        """
        Args:
            image_size (list, optional): m x n pixels of the output images as [width, height]. Defaults to [250,250].
            pixel_size (list, optional): Physical size of the pixels in mm. Defaults to [.25,.25].
            focal_length (list, optional): Focal length of the camera in mm. Defaults to [4100,4100].
            sequence_type (str, optional): "list" if supplying a set of <PPA&PSA> or "random" to generate a set. Defaults to "list".
            ppa_list (list, optional): If "sequence_type" is "list", a list of PPAs to generate projections - zipped with psa_list. Defaults to [0,20].
            psa_list (list, optional): If "sequence_type" is "list", a list of PSAs to generate projections - zipped with ppa_list. Defaults to [0,0].
            ppa_range (list, optional): If "sequence_type" is "random", the range of PPAs to generate projections from. Defaults to [-20,20].
            psa_range (list, optional): If "sequence_type" is "random", the range of PSAs to generate projections from. Defaults to [0,0].
            seed (int, optional): If "sequence_type" is "random", the random seed for the generator. 0 seeds from system clock. Defaults to 0.
            n_images (int, optional): If "sequence_type" is "random", the number of images to generate. Defaults to 2.
        """  
		self.image_size = image_size
		self.pixel_size = pixel_size
		self.focal_length = focal_length

		self.sequence_type = sequence_type # either "list" or "random"
		self.ppa_list = ppa_list
		self.psa_list = psa_list
		self.ppa_range = ppa_range
		self.psa_range = psa_range
		self.seed = seed
		self.n_images = n_images

		if sequence_type == "list":
			assert len(ppa_list) == len(psa_list), f"'ppa_list' dimension must match 'psa_list': {len(ppa_list)} vs {len(psa_list)}"
			self.n_images = len(ppa_list)
		elif sequence_type == "random":
			#TODO: implement random generation using seed
			pass

	def _instantiate_pyrender_scene(self):
		scene = pyrender.Scene()
		renderer = pyrender.OffscreenRenderer(self.image_size[0],self.image_size[1])

		light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
		cam = pyrender.IntrinsicsCamera(
				fx=self.focal_length[0],
				fy=self.focal_length[0],
				cx=self.image_size[0]/2,
				cy=self.image_size[1]/2,
				zfar=100000000000000 # `Infinite` clipping
			)

		light_obj = scene.add(light)
		cam_obj = scene.add(cam)

		return scene, renderer, light_obj, cam_obj

	def __call__(self, mesh):
		scene, renderer, light_obj, cam_obj = self._instantiate_pyrender_scene()

		mesh_obj = scene.add(mesh)

		imgs = []
		for i in range(self.n_images):
			pose = get_matrix(self.ppa_list[i], self.psa_list[i], self.pixel_size[0]*self.focal_length[0])
			imgs.append(
				self._get_one_image(pose, scene, renderer, light_obj, cam_obj)
			)

		scene.remove_node(mesh_obj)

		return np.dstack(imgs)

	def _get_one_image(self, pose, scene, renderer, light_obj, cam_obj):
		scene.set_pose(light_obj, pose)
		scene.set_pose(cam_obj, pose)

		image_rgb, depth = renderer.render(scene)

		# Convert to single channel (greyscale)
		image = np.array(image_rgb[:, :, 0])*.299 + np.array(image_rgb[:, :, 1])*.587 + np.array(image_rgb[:, :, 2])*.114
		return image.astype(np.uint8)  


def main():
    root_path = Path("../CoronaryVesselGeneration/AngioGenAppNew/output/testing/")

    keys = range(100)
    values = [str(root_path / f"{(i+1):04}" / "mesh.stl") for i in keys]

    meshes = dict(zip(keys, values))
    input()
    dataset = MeshDataset(meshes)
    input()

    dataloader = DataLoader(dataset, batch_size=100, num_workers=0)

    # Evaluate how long this all takes
    import time
    from PIL import Image

    for batch in dataloader:
        print(batch.shape)
        for i in range(3):
            im = Image.fromarray(batch.numpy()[i,:,:,0].astype(np.uint8),'L')
            file_name = f'{i:03d}.png'

            im.save(file_name)

    t1 = time.time()
    for batch in dataloader:
        t2 = time.time()
        input(t2-t1) # Allows memory to be checked externally
        t1 = time.time()

if __name__ == "__main__":
    main()