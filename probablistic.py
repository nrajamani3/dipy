import dipy
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,auto_response)
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking,CmcTissueClassifier,ParticleFilteringTracking)
from dipy.tracking.streamline import Streamlines
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy
from dipy.data import (default_sphere,read_stanford_pve_maps,read_stanford_labels)
from dipy.direction import ProbabilisticDirectionGetter
from dipy.io.streamline import save_trk
from dipy.viz import window,actor
from dipy.viz.colormap import line_colors
import nibabel as nib
import time
import numpy as np
import os
import matplotlib.pyplot as plt


interactive = False
renderer = window.Renderer()
start_time = time.time()
filename   = os.path.join("/home/nrajamani/Downloads/DIPYDATA/sub-control201/ses-01/dwi/","data.nii.gz")
img_to_use = dipy.data.load(filename)
data = img_to_use.get_data()
affine = img_to_use.affine

#import wm masks
label_filename_wm = os.path.join("/home/nrajamani/Downloads/DIPYDATA/sub-control201/ses-01/dwi/","pve_wm.nii.gz")
labels_img_wm = dipy.data.load(label_filename_wm)
labels_wm = labels_img_wm.get_data()
shape=labels_wm.shape
# import gm masks
label_filename_gm = os.path.join("/home/nrajamani/Downloads/DIPYDATA/sub-control201/ses-01/dwi/","pve_gm.nii.gz")
labels_img_gm = dipy.data.load(label_filename_gm)
labels_gm = labels_img_gm.get_data()
shape=labels_gm.shape
#import csf masks 
label_filename_csf = os.path.join("/home/nrajamani/Downloads/DIPYDATA/sub-control201/ses-01/dwi/","pve_csf.nii.gz")
labels_img_csf = dipy.data.load(label_filename_csf)
labels_csf = labels_img_csf.get_data()
shape=labels_csf.shape

bvals = os.path.join("/home/nrajamani/Downloads/DIPYDATA/sub-control201/ses-01/dwi","bvals")
bvecs = os.path.join("/home/nrajamani/Downloads/DIPYDATA/sub-control201/ses-01/dwi","bvecs")
gtab = gradient_table(bvals,bvecs)

#gtab = os.path.join("/home/nrajamani/Downloads/HNU1/data/","gtab.bval")
#gtab = dipy.data.load(gtab)
#affine = hardi_img.affine


response,ratio = auto_response(gtab,data,roi_radius=10,fa_thr=0.7)
csa_model = ConstrainedSphericalDeconvModel(gtab, response)
csa_fit = csa_model.fit(data, mask=labels_wm)
dg= ProbabilisticDirectionGetter.from_shcoeff(csa_fit.shm_coeff,max_angle=20.,sphere=default_sphere)

#Continous Map Criterion and Anatomically Constrainted Tractography(ACT) BOTH USES PVEs information from anatomical images to determine when the tractography stops.
#Both tssue classifiers use a trilinear interpolation at the tracing position CMC tissue classifier uses a probability derived frm the PVE maps to determine if the 
#streamline reaches a 'valid' or 'invalid' region.ACT uses a fixed threshold on the PVE maps. Both tissue classifiers used in conjuction with PFT. 

voxel_size=1
#avg_vox_size = np.average(voxel_size)
step_size = 0.2
cmc_classifier = CmcTissueClassifier.from_pve(labels_img_wm.get_data(),labels_img_gm.get_data(),labels_img_csf.get_data(),step_size=step_size,average_voxel_size=voxel_size)

seed_mask = labels_wm 
seeds = utils.seeds_from_mask(seed_mask, density=3, affine=affine)

pft_streamline_generator = ParticleFilteringTracking(dg,cmc_classifier,seeds,affine,max_cross=1,step_size=step_size,maxlen=1000,pft_back_tracking_dist=2,pft_front_tracking_dist=1,particle_count=15,return_all=False)
prob_streamline_generator = LocalTracking(dg,cmc_classifier,seeds,affine,max_cross=1,step_size=step_size,maxlen=1000,return_all=False)
#stream;omes = list(pft_strealine_generator)
streamlines = Streamlines(pft_streamline_generator)
save_trk("pft_streamline.trk",streamlines,affine,shape)
renderer.clear()
renderer.add(actor.line(streamlines,line_colors(streamlines)))
window.record(renderer, out_path='pft_streamlines_3.png', size=(600,600))
print("------%s seconds -----" % (time.time() - start_time))














































#gfa = csa_model.fit(data, mask=white_matter).gfa
#classifier = ThresholdTissueClassifier(gfa, .25)
#fod = csd_fit.odf(small_sphere)
#pmf = fod.clip(min=0)
#prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30.,
#                                                sphere=small_sphere)
#streamlines_generator = LocalTracking(prob_dg, classifier, seeds, affine, step_size=.5)
#save_trk("probabilistic_small_sphere.trk", streamlines_generator, affine, labels.shape)
#prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
#                                                    max_angle=30.,
#                                                    sphere=default_sphere)
#streamlines_generator = LocalTracking(prob_dg, classifier, seeds, affine, step_size=.5)
#save_trk("probabilistic_shm_coeff.trk", streamlines_generator, affine, labels.shape)

##probablistic 23.4742641449 seconds###

