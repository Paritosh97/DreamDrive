from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images_resized" # "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False

        # add by deformable gs
        self.load2gpu_on_the_fly = False
        self.is_blender = False
        self.is_6dof = False
        
        # add by hexplane gs
        self.net_width = 64
        self.timebase_pe = 4
        self.defor_depth = 1
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.6
        self.plane_tv_weight = 0.0001
        self.time_smoothness_weight = 0.01
        self.l1_time_planes = 0.0001
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]
                            }
        self.multires = [1, 2, 4, 8]
        self.no_dx=False
        self.no_grid=False
        self.no_ds=True 
        self.no_dr=True
        self.no_do=True
        self.no_dshs=False
        self.feat_head=False # default is True, but change to False temporarily
        self.empty_voxel=False
        self.grid_pe=0
        self.static_mlp=False
        self.apply_rotation=False

        # for semantic features
        self.use_semantic_features = True ####
        self.semantic_feature_key = "dino_reg_pca32" # "dino_reg_pca32" # "in_feats_0_pca32"
        self.novel_view_key = "stop"
        self.dynamic_th = 0.3 # image norm threshold for training dynamic model
        self.cluster_th = 0.15 # clustering threshold, important! Optimal 0.15

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        """self.iterations = 30_000"""
        self.iterations = 33_000
        self.warm_up = 3_000
        self.position_lr_init =  0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.beta_lr = 0.005 # add beta lr
        self.sky_lr = 0.001
        self.affine_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2

        # desification
        self.densification_interval = 100
        self.opacity_reset_interval = 300000 # never, originally 3000
        """self.densify_from_iter = 4000"""
        self.densify_from_iter = 7000
        """self.densify_until_iter = 20000 #-1 # nodensify # 15_000"""
        self.densify_until_iter = 23000 #-1 # nodensify # 15_000
        self.densify_grad_threshold = 0.0007
        self.random_background = False

        # add by hexplane gs
        self.coarse_iterations = 5000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        
        self.lambda_feat = 0.001
        self.dx_reg = False
        self.lambda_dx = 0.001
        self.lambda_dshs = 0.001
        self.lambda_depth = 0.5

        self.deformation_lr_init = 0.000016
        self.deformation_lr_final = 0.0000016
        self.deformation_lr_delay_mult = 0.01
        
        self.grid_lr_init = 0.00016
        self.grid_lr_final = 0.000016

        # for dynamic model
        """self.static_iterations = 3000 # num steps for training the static model"""
        self.static_iterations = 6000 # num steps for training the static model
        self.dynamic_iterations = 3_000 # num steps for training the dynamic model
        self.dynamic_net_lr =  0.016 
        self.dynamic_net_lr_final = 0.0000016
        self.dynamic_net_lr_delay_mult = 0.01
        self.dynamic_net_lr_max_steps = self.dynamic_iterations # self.iterations # 
        self.dynamic_reg_coef = 0.5
        
        self.cluster_lambda = 1.0
        
        # for deformation net
        self.deform_lr_init =  0.000016
        self.deform_lr_final = 0.0000016
        self.deform_lr_delay_mult = 0.01
        self.deform_lr_max_steps = 30_000


        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
