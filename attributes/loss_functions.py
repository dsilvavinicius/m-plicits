import torch
import torch.nn.functional as F

import numpy as np

import diff_operators

#import trimesh

def sdf_constraint_on_surf(gt_sdf, pred_sdf):
   return torch.where(gt_sdf == 0, pred_sdf ** 2, torch.zeros_like(pred_sdf))

def sdf_constraint_off_surf(gt_sdf, pred_sdf):
   return torch.where(gt_sdf != 0, (gt_sdf - pred_sdf) ** 2, torch.zeros_like(pred_sdf))

def vector_aligment_on_surf(gt_sdf, gt_vectors, pred_vectors):
   return torch.where(gt_sdf == 0, 1 - F.cosine_similarity(pred_vectors, gt_vectors, dim=-1)[..., None], torch.zeros_like(gt_sdf))

def direction_aligment_on_surf(gt_sdf, gt_dirs, pred_dirs):
   return torch.where(gt_sdf == 0, 1 - (F.cosine_similarity(pred_dirs, gt_dirs, dim=-1)[..., None])**2, torch.zeros_like(gt_sdf))
    
def eikonal_constraint(gradient):
   return ((gradient.norm(dim=-1) - 1.) ** 2).unsqueeze(-1)


def vector_dot(u, v):
    return torch.sum(u * v, dim=-1, keepdim=True)

def vector_normalize(v):
    v_norm = torch.norm(v, dim=-1).unsqueeze(-1)
    return v/v_norm

def true_sdf(model_output, gt):
    '''Uses true SDF value off surface.
    x: batch of input coordinates
    y: usually the output of the trial_soln function
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    
    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)
    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
            'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
            'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() *1e2 ,#* 1e1,
            'grad_constraint': eikonal_constraint(gradient).mean() * 5e1}#1e1}


# learn a second model that add details to a previously trained model
class loss_add_detail(torch.nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        # Define the model.
        self.model = trained_model
        self.model.cuda()

    def forward(self, model_output, gt):

         gt_sdf = gt['sdf']
         gt_normals = gt['normals']
         
         #local
         coords = model_output['model_in']
         pred_sdf = model_output['model_out']

         trained_model = self.model(coords)
         trained_coords = trained_model['model_in']
         trained_sdf = trained_model['model_out']

         gradient = diff_operators.gradient(pred_sdf, coords) + diff_operators.gradient(trained_sdf, trained_coords)
         
         # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
         return {'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf + trained_sdf).mean() * 3e3,
                  'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf + trained_sdf).mean() * 2e2,
                  'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() *1e2 ,#* 1e1,
                  'grad_constraint': eikonal_constraint(gradient).mean() * 5e1}#1e1}


#learn the mean curvature of a neural surface
class loss_mean_curvature(torch.nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        # Define the model.
        self.model = trained_model
        self.model.cuda()

    def forward(self, model_output, gt):
   
         coords = model_output['model_in']
         pred_curvature = model_output['model_out']

         trained_model = self.model(coords)
         global_coords = trained_model['model_in']
         global_sdf    = trained_model['model_out']

         # ground truth mean curvature
         curvature = diff_operators.mean_curvature(global_sdf, global_coords)

         constraint = (curvature - pred_curvature)**2

         return {'constraint': constraint.mean()}


#learn the solution of PDEs on a neural surface
class loss_PDE_on_surface(torch.nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        # Define the model.
        self.model = trained_model
        self.model.cuda()

    def forward(self, model_output, gt):
        #return self.loss_laplace(model_output, gt)
        #return self.loss_laplace_test(model_output, gt)
        return self.loss_reaction_diffusion(model_output, gt)
    
    
    def loss_curvature_4D(self, model_output, gt):
   
         coords_4d = model_output['model_in']
         pred_curvature = model_output['model_out']

         coords_3d = coords_4d[..., :3]
         trained_model = self.model(coords_3d)
         coords_3d = trained_model['model_in']
         sdf = trained_model['model_out']

         # ground truth mean curvature
         curvature = diff_operators.mean_curvature(sdf, coords_3d)

         diff_constraint = (curvature - pred_curvature)**2

         #constant along the gradient
         pred_curvature_grad  = diff_operators.gradient(pred_curvature, coords_4d)[...,:3]
         
         grad = diff_operators.gradient(sdf, coords_3d)
         grad_constraint   = torch.sum(grad * pred_curvature_grad, dim=-1, keepdim=True)**2

         return { 'diff_constraint': diff_constraint.mean(),#* 1e1,
                  'grad_constraint': grad_constraint.mean()*0.1,
                  }#1e1}

    def loss_laplace(self, model_output, gt):

         coords_4d = model_output['model_in']
         pred_function = model_output['model_out']

         grad  = diff_operators.gradient(pred_function, coords_4d)
         ut = grad[...,3].unsqueeze(-1)
    
         grad = grad[...,:3]
         
         #project grad on the tangent space
         coords_3d = coords_4d[..., :3]
         sdf_model = self.model(coords_3d)
         coords_3d = sdf_model['model_in']
         sdf = sdf_model['model_out']
         sdf_grad = diff_operators.gradient(sdf, coords_3d).detach()

        # if trained_model is not a sdf
        #  sdf_grad_norm = torch.norm(sdf_grad, dim=-1).unsqueeze(-1)
        #  unit_sdf_grad = sdf_grad/sdf_grad_norm
         
        #  proj_grad = grad - unit_sdf_grad*vector_dot(grad, unit_sdf_grad)
        #  div = diff_operators.divergence(proj_grad*sdf_grad_norm, coords_4d)
        #  laplace_constraint = (ut - div/sdf_grad_norm)**2
         
        # if trained_model is a sdf 
         proj_grad = grad - sdf_grad*vector_dot(grad, sdf_grad)
         div = diff_operators.divergence(proj_grad, coords_4d)
         #laplace_constraint = (ut - 0.001*div)**2
         laplace_constraint = (ut - 0.002*div)**2
         
         # ground truth mean curvature
         curvature = diff_operators.mean_curvature(sdf, coords_3d)
         time = coords_4d[...,3].unsqueeze(-1)
         diff_constraint = torch.where(time==0, (curvature - pred_function)**2, torch.zeros_like(curvature))

         #constant along the gradient
         grad_constraint = vector_dot(grad, sdf_grad)**2
         grad_constraint = torch.where(time==0, grad_constraint, torch.zeros_like(curvature))
         
         laplace_constraint = torch.where(time!=0, laplace_constraint, torch.zeros_like(curvature))

         return { 'diff_constraint': diff_constraint.mean()*1e1,#* 1e1,
                  'grad_constraint': grad_constraint.mean(),
                  'laplace_constraint': laplace_constraint.mean(),
                }#1e1}

    def loss_laplace_test(self, model_output, gt):

         gt_textures = gt['textures']

         coords_4d = model_output['model_in']
         pred_function = model_output['model_out']

         grad  = diff_operators.gradient(pred_function, coords_4d)
         ut = grad[...,3].unsqueeze(-1)
    
         grad = grad[...,:3]
         
         #project grad on the tangent space
         coords_3d = coords_4d[..., :3]
         sdf_model = self.model(coords_3d)
         coords_3d = sdf_model['model_in']
         sdf = sdf_model['model_out']
         sdf_grad = diff_operators.gradient(sdf, coords_3d).detach()
         
        # if trained_model is a sdf 
         proj_grad = grad - sdf_grad*vector_dot(grad, sdf_grad)
         div = diff_operators.divergence(proj_grad, coords_4d)
         #laplace_constraint = (ut - 0.001*div)**2
         laplace_constraint = (ut - 0.002*div)**2
         
         # ground truth mean curvature
         time = coords_4d[...,3].unsqueeze(-1)
         diff_constraint = torch.where(time==0, (gt_textures - pred_function)**2, torch.zeros_like(gt_textures))

         #constant along the gradient
         grad_constraint = vector_dot(grad, sdf_grad)**2
         grad_constraint = torch.where(time==0, grad_constraint, torch.zeros_like(gt_textures))
         
         laplace_constraint = torch.where(time!=0, laplace_constraint, torch.zeros_like(gt_textures))

         return { 'diff_constraint': diff_constraint.mean()*1e1,#* 1e1,
                  'grad_constraint': grad_constraint.mean(),
                  'laplace_constraint': laplace_constraint.mean(),
                }#1e1}

    def loss_reaction_diffusion(self, model_output, gt):

         #gt_textures = gt['textures']
         coords_4d = model_output['model_in']
         pred_function = model_output['model_out']
         
         u1 = pred_function[...,0].unsqueeze(-1)
         u2 = pred_function[...,1].unsqueeze(-1)
         
         grad_u1  = diff_operators.gradient(u1, coords_4d)
         grad_u2  = diff_operators.gradient(u2, coords_4d)
         
         u1t = grad_u1[...,3].unsqueeze(-1)
         u2t = grad_u2[...,3].unsqueeze(-1)
    
         grad_u1 = grad_u1[...,:3]
         grad_u2 = grad_u2[...,:3]
         
         #project grad on the tangent space
         coords_3d = coords_4d[..., :3]
         sdf_model = self.model(coords_3d)
         coords_3d = sdf_model['model_in']
         sdf = sdf_model['model_out']
         sdf_grad = diff_operators.gradient(sdf, coords_3d).detach()
         
         pred_curvature = diff_operators.mean_curvature(sdf, coords_3d)
         u1_0 = torch.ones_like(u1)
         u2_0 = torch.tanh(torch.relu(pred_curvature-12.0))
         
        # if trained_model is a sdf 
         proj_grad_u1 = grad_u1 - sdf_grad*vector_dot(grad_u1, sdf_grad)
         div_u1 = diff_operators.divergence(proj_grad_u1, coords_4d)
         
         proj_grad_u2 = grad_u2 - sdf_grad*vector_dot(grad_u2, sdf_grad)
         div_u2 = diff_operators.divergence(proj_grad_u2, coords_4d)

         feed = 0.0782
         kill = 0.06
         D1 = 1.0
         D2 = 0.5

         F = - u1*u2*u2 + feed*(1. - u1)
         G =   u1*u2*u2 - (feed + kill)*u2
    
         eq1_constraint = (u1t - 0.1*(F + D1*div_u1))**2
         eq2_constraint = (u2t - 0.1*(G + D2*div_u2))**2
         
         time = coords_4d[...,3].unsqueeze(-1)
         diff1_constraint = torch.where(time==0, (u1_0 - u1)**2, torch.zeros_like(u1))
         diff2_constraint = torch.where(time==0, (u2_0 - u2)**2, torch.zeros_like(u2))

         #constant along the gradient
         grad_constraint = vector_dot(grad_u1, sdf_grad)**2 + vector_dot(grad_u2, sdf_grad)**2
         grad_constraint = torch.where(time==0, grad_constraint, torch.zeros_like(u1))
         
         eq1_constraint = torch.where(time!=0, eq1_constraint, torch.zeros_like(u1))
         eq2_constraint = torch.where(time!=0, eq2_constraint, torch.zeros_like(u2))

         return { 'diff1_constraint': diff1_constraint.mean()*1e2,
                  'diff2_constraint': diff2_constraint.mean()*1e2,
                  'grad_constraint': grad_constraint.mean(),
                  'eq1_constraint': eq1_constraint.mean()*1e1,
                  'eq2_constraint': eq2_constraint.mean()*1e1,
                }


#learn the solution of PDEs on a neural surface
class loss_SDF_on_surface(torch.nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        # Define the model.
        self.model = trained_model
        self.model.cuda()

    def forward(self, model_output, gt):
        return self.eikonal(model_output, gt)
    

    def eikonal(self, model_output, gt):
         gt_distances = gt['distances'].unsqueeze(-1)

         coords = model_output['model_in']
         pred_sdf_on_surface = model_output['model_out']

         grad  = diff_operators.gradient(pred_sdf_on_surface, coords)
         
         #project grad on the tangent space
         model_input = {"coords":coords}
         sdf_model = self.model(model_input)
         coords_sdf = sdf_model['model_in']
         space_sdf = sdf_model['model_out']
         sdf_grad = diff_operators.gradient(space_sdf, coords_sdf).clone().detach()

        # if trained_model is a sdf 
         proj_grad = grad - sdf_grad*vector_dot(grad, sdf_grad)

         eikonal_constraint = (vector_dot(proj_grad,proj_grad) - 1)**2
         
        #  test = torch.abs(coords[..., 0]-0.3).unsqueeze(-1)<0.005
        #  on_curve_constraint = torch.where(test, pred_sdf_on_surface**2, torch.zeros_like(pred_sdf_on_surface))
        #  #off_curve_constraint = torch.where(test, torch.zeros_like(pred_sdf_on_surface), torch.exp(-100*pred_sdf_on_surface**2))
        #  off_curve_constraint  = torch.where((coords[..., 0]-0.3).unsqueeze(-1)<-0.005, torch.exp( 100*pred_sdf_on_surface), torch.zeros_like(pred_sdf_on_surface))
        #  off_curve_constraint += torch.where((coords[..., 0]-0.3).unsqueeze(-1)> 0.005, torch.exp(-100*pred_sdf_on_surface), torch.zeros_like(pred_sdf_on_surface))
         
        #  #alight between the curve normal
        #  normals = torch.zeros_like(coords)
        #  normals[..., 0] += 1
        #  proj_normals = (normals - sdf_grad*vector_dot(normals, sdf_grad)).clone().detach()
        #  normal_constraint = torch.where(test, 1 - F.cosine_similarity(proj_normals, grad, dim=-1)[..., None], torch.zeros_like(pred_sdf_on_surface))

        #visualize data
         visualize = False
         if visualize:
            on_curv = coords[test, ...]
            off_curv = coords[torch.abs(coords[..., 0]-0.3)>=0.01,...]
            on_colors = torch.zeros((on_curv.size(0), 3))
            off_colors = torch.zeros((off_curv.size(0), 3))
            off_colors[...,0] = 1
            
            coords =  torch.cat((on_curv, off_curv), axis=0).cpu().detach().numpy()
            colors =  torch.cat((on_colors, off_colors), axis=0).cpu().detach().numpy()
            trimesh.points.PointCloud(vertices = coords, colors=colors).show()

         distance_constraint = (gt_distances - pred_sdf_on_surface)**2

        #constant along the gradient
         grad_constraint = vector_dot(grad, sdf_grad)**2
        
         return { #'on_curve_constraint': on_curve_constraint.sum()*1e4,#* 1e1,
                  #'off_curve_constraint': off_curve_constraint.mean()* 1e4,
                  'distance_constraint': distance_constraint.mean()*1e3,
                 # 'eikonal_constraint': eikonal_constraint.mean()*1e2,
                  #'normal_constraint': normal_constraint.sum()*1e1,
                 # 'grad_constraint': grad_constraint.mean()*1e2,
                }#1e1}



# learn a second model that add texture to a previously trained model
class loss_texture(torch.nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        # Define the model.
        self.model = trained_model
        self.model.cuda()

    def forward(self, model_output, gt):
       #return self.loss_normals(model_output, gt)
       return self.loss_texture(model_output, gt)
    
    def loss_normals(self, model_output, gt): 
         gt_sdf = gt['sdf']
         gt_normals = gt['normals']
         
         pred_texture = model_output['model_out']

         normal_constraint =  torch.where(gt_sdf == 0, 1 - F.cosine_similarity(pred_texture, gt_normals, dim=-1)[..., None], torch.zeros_like(gt_sdf))
         
         return { 'normal_constraint': normal_constraint.mean(),#* 1e1,
                  'grad_constraint': eikonal_constraint(pred_texture).mean() * 5e1
                  }#1e1}

    def loss_texture(self, model_output, gt): 
         #gt_sdf = gt['sdf']
         gt_textures = gt['textures']

         coords = model_output['model_in']
         pred_texture = model_output['model_out']
         pred_texture_x = pred_texture[...,0]
         pred_texture_y = pred_texture[...,1]
         pred_texture_z = pred_texture[...,2]
         texture_grad_x  = diff_operators.gradient(pred_texture_x, coords)
         texture_grad_y  = diff_operators.gradient(pred_texture_y, coords)
         texture_grad_z  = diff_operators.gradient(pred_texture_z, coords)

         texture_constraint = (gt_textures - pred_texture)**2
         
        #constant along the gradient 
         trained_model = self.model(coords)
         trained_coords = trained_model['model_in']
         trained_sdf    = trained_model['model_out']
         
         sdf_grad = diff_operators.gradient(trained_sdf, trained_coords)
         diff_constraint   = torch.sum(sdf_grad * texture_grad_x, dim=-1, keepdim=True)**2
         diff_constraint  += torch.sum(sdf_grad * texture_grad_y, dim=-1, keepdim=True)**2
         diff_constraint  += torch.sum(sdf_grad * texture_grad_z, dim=-1, keepdim=True)**2

         return { 'texture_constraint': texture_constraint.mean()*1e3,#* 1e1,
                  #'diff_constraint': diff_constraint.mean()* 1e-1,
                  }#1e1}


def loss_curvatures(model_output, gt):
    '''Uses true SDF value off surface and tries to fit gaussian curvatures on
    the 0 level-set.

    x: batch of input coordinates
    y: usually the output of the trial_soln function
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_min_curvature = gt["min_curvatures"]
    gt_max_curvature = gt["max_curvatures"]
    gt_dirs = gt["max_principal_directions"]

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)
    hessian = diff_operators.hessian(pred_sdf, coords)
    pred_dirs = diff_operators.principal_directions(gradient, hessian[0])

    dirs_constraint = direction_aligment_on_surf(gt_sdf, gt_dirs, pred_dirs[0][...,0:3])

    aux_dirs_constraint = torch.where(gt_sdf == 0, F.cosine_similarity(pred_dirs[0][...,0:3], gt_normals, dim=-1)[..., None]**2, torch.zeros_like(gt_sdf))

    dirs_constraint = dirs_constraint + 0.1*aux_dirs_constraint

   #removing umbilical points of the pred sdf
    #dirs_constraint = torch.where(pred_dirs[0][...,3].unsqueeze(-1) == 0, dirs_constraint, torch.zeros_like(dirs_constraint))

    #removing problematic curvatures and planar points
    planar_curvature = 0.5*torch.abs(gt_min_curvature-gt_max_curvature)
    dirs_constraint = torch.where(planar_curvature > 10  , dirs_constraint, torch.zeros_like(dirs_constraint))
    dirs_constraint = torch.where(planar_curvature < 5000, dirs_constraint, torch.zeros_like(dirs_constraint))

    return {'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
            'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
            'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() *1e2,#* 1e1,
            'grad_constraint': eikonal_constraint(gradient).mean() * 5e1,
            'dirs_constraint': dirs_constraint.mean()
            }


def true_sdf_curvature(model_output, gt):
    '''Uses true SDF value off surface and tries to fit gaussian curvatures on
    the 0 level-set.

    x: batch of input coordinates
    y: usually the output of the trial_soln function
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    gt_curvature = gt["curvature"]

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)

   # mean curvature
    pred_curvature = diff_operators.mean_curvature(pred_sdf, coords)
    curvature_diff = torch.tanh(100*pred_curvature) - torch.tanh(100*gt_curvature)

    #consider the curature constraint only on the surface
    curv_constraint = torch.where(gt_sdf == 0, curvature_diff ** 2, torch.zeros_like(pred_curvature))
    #remove problematic curvatures and planar points
    curv_constraint = torch.where(torch.abs(gt_curvature) < 5000, curv_constraint, torch.zeros_like(pred_curvature))
    curv_constraint = torch.where(torch.abs(gt_curvature) > 10, curv_constraint, torch.zeros_like(pred_curvature))

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    return {'sdf_on_surf': sdf_constraint_on_surf(gt_sdf, pred_sdf).mean() * 3e3,
            'sdf_off_surf': sdf_constraint_off_surf(gt_sdf, pred_sdf).mean() * 2e2,
            'normal_constraint': vector_aligment_on_surf(gt_sdf, gt_normals, gradient).mean() *1e2 ,#* 1e1,
            'grad_constraint': eikonal_constraint(gradient).mean() * 5e1,
            'curv_constraint': curv_constraint.mean() * 5 }


def reaction_diffusion_test(model_output, gt):
    
    gt_u = gt['function']

    x = model_output['model_in']  # (meta_batch_size, num_points, 3)
    u = model_output['model_out']  # (meta_batch_size, num_points, 1)
    
    u1 = u[...,0].unsqueeze(-1)
    u2 = u[...,1].unsqueeze(-1)
    
    grad_u1  = diff_operators.gradient(u1, x)
    grad_u2  = diff_operators.gradient(u2, x)
    
    u1t = grad_u1[...,2].unsqueeze(-1)
    u1x = grad_u1[...,0].unsqueeze(-1)
    u1y = grad_u1[...,1].unsqueeze(-1)

    u2t = grad_u2[...,2].unsqueeze(-1)
    u2x = grad_u2[...,0].unsqueeze(-1)
    u2y = grad_u2[...,1].unsqueeze(-1)

    grad_u1 = grad_u1[...,:2]
    grad_u2 = grad_u2[...,:2]
    
    u1_0 = gt_u[...,0].unsqueeze(-1)
    u2_0 = gt_u[...,1].unsqueeze(-1)    

# if trained_model is a sdf 
    div_u1 = diff_operators.divergence(grad_u1, x)
    div_u2 = diff_operators.divergence(grad_u2, x)

    feed = 0.055
    kill = 0.062
    D1 = 8.0e-06
    D2 = D1/2

    F = - u1*u2*u2 + feed*(1. - u1)
    G =   u1*u2*u2 - (feed + kill)*u2

    #eq2_constraint = (u2t - (grad_u2[...,0].unsqueeze(-1)+grad_u2[...,1].unsqueeze(-1)))**2 #transport along the diagonal

    eq1_constraint = (u1t - 200.0*(F + D1*div_u1))**2
    eq2_constraint = (u2t - 200.0*(G + D2*div_u2))**2
    
    diff1_constraint = torch.where(u1_0 != -1000, (u1_0 - u1)**2, torch.zeros_like(u1_0))
    diff2_constraint = torch.where(u2_0 != -1000, (u2_0 - u2)**2, torch.zeros_like(u2_0))
    
    x1 = x[...,0].unsqueeze(-1)
    x2 = x[...,1].unsqueeze(-1)
    
    derivative1_constraint = torch.where(u1_0 != -1000, u1x**2 + u1y**2, torch.zeros_like(u1_0))
    initial_condition = torch.exp(-(x1**2)/0.01 - (x2**2)/0.01)
    derivative2_constraint = torch.where(u2_0 != -1000, (u2x - initial_condition*(-2.0*x1/0.01))**2 + (u2y - initial_condition*(-2.0*x2/0.01))**2, torch.zeros_like(u2_0))
    
    circ = torch.maximum(torch.abs(x[...,0].unsqueeze(-1)), torch.abs(x[...,1].unsqueeze(-1))).clone().detach()
    boundary_constraint = torch.where(circ > 0.9, (u1-1)**2 + u2**2 + u1t**2 + u2t**2,  torch.zeros_like(u1) )
    #boundary_constraint = torch.where(u1_0 == -1000, boundary_constraint,  torch.zeros_like(u1) )

    return { 'diff1_constraint': diff1_constraint.sum()*1e1,
        'derivative1_constraint': derivative1_constraint.sum(),#*1e-1,
        'diff2_constraint': diff2_constraint.sum()*1e1,
        'derivative2_constraint': derivative2_constraint.sum(),#*1e-1,
        'eq1_constraint': eq1_constraint.mean(),
        'eq2_constraint': eq2_constraint.mean(),
        'boundary_constraint': boundary_constraint.sum(),
    }

