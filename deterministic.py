import dipy
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,auto_response)
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy
from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.streamline import save_trk
from dipy.tracking.streamline import Streamlines
from dipy.viz import window,actor
from dipy.viz.colormap import line_colors
import nibabel as nib
import time
import numpy as np
import os
import matplotlib.pyplot as plt


interactive = False
start_time = time.time()
filename   = os.path.join("/home/nrajamani/Downloads/DIPYDATA/sub-control201/ses-01/dwi/","data.nii.gz")
img_to_use = dipy.data.load(filename)
_,_, img_pve_wm = read_stanford_pve_maps()
data = img_to_use.get_data()
affine = img_to_use.affine
label_filename = os.path.join("/home/nrajamani/Downloads/HNU1/scripts/","shape_mask_mask.nii.gz")
labels_img = dipy.data.load(label_filename)
labels = labels_img.get_data()
bvals = os.path.join("/home/nrajamani/Downloads/DIPYDATA/sub-control201/ses-01/dwi","bvals")
bvecs = os.path.join("/home/nrajamani/Downloads/DIPYDATA/sub-control201/ses-01/dwi","bvecs")
gtab = gradient_table(bvals,bvecs)

#gtab = os.path.join("/home/nrajamani/Downloads/HNU1/data/","gtab.bval")
#gtab = dipy.data.load(gtab)
#affine = hardi_img.affine
seed_mask = labels
seeds = utils.seeds_from_mask(seed_mask, density=5, affine=affine)

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.5)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)
csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
csd_fit = csd_model.fit(data, mask=labels)

dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,max_angle=80.,sphere=default_sphere)

tensor_model = dti.TensorModel(gtab)
tenfit = tensor_model.fit(data, mask=labels>0)

FA = fractional_anisotropy(tenfit.evals)
threshold_classifier = ThresholdTissueClassifier(FA, .5)


fig = plt.figure()
mask_fa = FA.copy()
mask_fa[mask_fa < 0.5] = 0
#p#lt.xticks([])
#plt.yticks([])
#plt.imshow(mask_fa[:, :, data.shape[2] // 2].T, cmap='gray', origin='lower',
#           interpolation='nearest')
#fig.tight_layout()
#fig.savefig('threshold_fa_3_0.5.png')


all_streamlines_threshold_classifier = LocalTracking(dg,threshold_classifier,seeds,affine,step_size=0.4,return_all=True)
save_trk("deterministic_threshold_classifier_all.trk",all_streamlines_threshold_classifier,affine,labels.shape)
streamlines = Streamlines(all_streamlines_threshold_classifier)
ren = window.Renderer()
if window.have_vtk:
	window.clear(ren)
	ren.add(actor.line(streamlines,line_colors(streamlines)))
	window.record(ren,out_path='/home/nrajamani/Downloads/HNU1/scripts/density_vs_exec/all_streamlines_threshold_classifier_4.png',size=(600,600))
	if interactive:
		window.show(ren)
print("----%s seconds -----" %(time.time()-start_time))

## [2,2,2] = execution time 849.603137016 seconds ##
## [3,3,3] = execution time  2183.9916029 seconds seconds##
## [4,4,4] = execution time  
### 5 = execution time 32
### 6 = 28.07976 seconds
### 7 = 27.07548
### 8 = 26.963033
### 9 = 25.994
### 10 = 24.00687
#Threshold classifier##
