import numpy as np

class MatrixConverter(object):
    @classmethod
    def DICOMtoExtrinsics(cls, positioner_primary_angle, positioner_secondary_angle, source_to_image_distance, table_displacement=None):
        '''
        positioner_primary_angle is primary positioner angle, positioner_secondary_angle is secondary
        '''
        anatomical_to_camera_transform = np.array([
            [1,0,0],
            [0,0,-1],
            [0,1,0]
        ])
        R1 = np.array([
            [np.cos(np.deg2rad(positioner_primary_angle)), -np.sin(np.deg2rad(positioner_primary_angle)), 0],
            [np.sin(np.deg2rad(positioner_primary_angle)), np.cos(np.deg2rad(positioner_primary_angle)), 0],
            [0, 0, 1]
        ])
        # secondary (C/C): CHECK
        R2 = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(positioner_secondary_angle)), -np.sin(np.deg2rad(positioner_secondary_angle))],
            [0, np.sin(np.deg2rad(positioner_secondary_angle)), np.cos(np.deg2rad(positioner_secondary_angle))]
        ])

        params = {}
        params["R"] = anatomical_to_camera_transform @ R1 @ R2

        params["t"] = np.array([0, 0.5*source_to_image_distance, 0])
        if table_displacement is not None:
            params["t"] = np.add(params["t"],np.array([table_displacement[0],table_displacement[1],table_displacement[2]]))

        params["t"] = anatomical_to_camera_transform @ params["t"]

        return params

    @classmethod
    def poseToExtrinsics(cls, P):
        '''
        Generate, set and return R and t (extrinsic and translation) given known pose
        '''
        raise NotImplementedError

    @classmethod
    def extrinsicstoPose(cls, R, t):
        '''
        Generate, set and return pose given known R and t (extrinsic and translation)
        '''
        rot = np.array([
            [ 1, 0, 0],
            [ 0,-1, 0],
            [ 0, 0,-1],
        ])

        R_cam_T = np.matmul(rot,R)
        T_cam = np.matmul(np.linalg.inv(R_cam_T), -rot).dot(t)

        P = np.zeros((4,4))
        P[:3,:3] = R_cam_T.transpose()
        P[:3,3] = T_cam.flatten()
        P[3,3]  = 1

        return P

    @classmethod
    def camParamsToIntrinsic(cls,xfov=None,yfov=None,image_dims=None,fx=None,fy=None):
        '''
        Generate, set and return K (intrinsic) given known yfov and image_dims
        '''
        if image_dims is None:
            raise Exception("camParamsToIntrinsics needs image_dims")
        if  xfov is None and yfov is None and fx is None and fy is None:
            raise Exception("camParamsToIntrinsics needs at least one of xfov, yfov, fx or fy")

        if fx is None and xfov is not None:
            fx = 0.5*image_dims[0]/(np.tan(0.5*xfov))
        if fy is None and yfov is not None:
            fy = 0.5*image_dims[1]/(np.tan(0.5*yfov))

        if fx is None:
            fx = fy
        if fy is None:
            fy = fx

        K = np.array([
            [fx, 0, image_dims[0]/2],
            [0, fy, image_dims[1]/2],
            [0, 0, 1]
        ])
        return K

    @classmethod
    def intrinsicToParams(cls,K):
        '''
        Generate, set and return fields of view and image_dims given known K (intrinsic)
        '''
        params = {}
        params["image_dims"] = (int(K[0,2]*2),int(K[1,2]*2))
        params["cx"] = int(K[0,2])
        params["cy"] = int(K[1,2])
        params["fx"] = K[0,0]
        params["fy"] = K[1,1]
        params["xfov"] = 2 * np.arctan(K[0,2] / K[0,0])
        params["yfov"] = 2 * np.arctan(K[1,2] / K[1,1])
        return params