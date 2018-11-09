#!/usr/bin/env python

# Copyright 2016 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# track.py
# Created by Will Gray Roncal on 2016-01-28.
# Email: wgr@jhu.edu

from __future__ import print_function

import numpy as np
import nibabel as nb
import dipy
import dipy.reconst.dti as dti
from dipy.reconst.dti import TensorModel, fractional_anisotropy, quantize_evecs
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier,LocalTracking)
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.tracking.streamline import Streamlines
from dipy.direction import peaks_from_model
from dipy.tracking.eudx import EuDX
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
from dipy.io.streamline import save_trk
from dipy.viz import window,actor
from dipy.viz.colormap import line_colors

#class track():

    #def __init__(self):
       # """
       # Tensor and fiber tracking class
       # """
        # WGR:TODO rewrite help text
     #   pass

def track(dwi_file,bval,bvec,mask_file,stop_val=0.1):
       # """
       # Tracking with basic tensors and basic eudx - experimental
       # We now force seeding at every voxel in the provided mask for
       # simplicity.  Future functionality will extend these options.
       # **Positional Arguments:**

       #         dwi_file:
       #             - File (registered) to use for tensor/fiber tracking
       #         mask_file:
       #             - Brain mask to keep tensors inside the brain
       #         gtab:
       #             - dipy formatted bval/bvec Structure

       # **Optional Arguments:**
       #         stop_val:
       #             - Value to cutoff fiber track
       # """

    #img = nb.load(dwi_file)
    #data = img.get_data()
    dwi = dipy.data.load(dwi_file)
    data = dwi.get_data()

    #img = nb.load(mask_file)
    #mask = img.get_data()
    dwi_mask = dipy.data.load(mask_file)
    mask = dwi_mask.get_data()
    gtab = gradient_table(bval,bvec)

    affine = dwi.affine 

    seed_mask = mask
    seeds = utils.seeds_from_mask(seed_mask, density=1,affine=affine)
        # use all points in mask
    seedIdx = np.where(mask > 0)  # seed everywhere not equal to zero                                     
    seedIdx = np.transpose(seedIdx)
    sphere = get_sphere('symmetric724')

    csd_model = ConstrainedSphericalDeconvModel(gtab,None,sh_order=6)
    csd_fit = csd_model.fit(data,mask=mask)


    tensor_model = dti.TensorModel(gtab)
    tenfit = tensor_model.fit(data,mask=mask)
    FA = fractional_anisotropy(tenfit.evals)
    classifier = ThresholdTissueClassifier(FA, 0.1)


    dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,max_angle=80.,sphere=sphere)

    streamlines_generator = LocalTracking(dg,classifier,seeds,affine,step_size = 0.5)
    streamlines = Streamlines(streamlines_generator)
    trk_file = save_trk("deterministic_threshold_DDG_samp_data.trk",streamlines,affine = affine,shape=mask.shape)

    #ren = window.Renderer()
    #if window.have_vtk:
    #    window.clear(ren)
    #    ren.add(actor.line(streamlines,line_colors(streamlines)))
    #    window.record(ren,out_path='/home/nrajamani/Downloads/HNU1/scripts/density_vs_exec/DDG.png',size=(600,600))
    #    if interactive:
     #       window.show(ren)
    #return trk_file

        #response,ratio = auto_response(gtab,data,roi_radius=10,fa_thr=0.2) 
        #model = TensorModel(gtab)
        #ten = model.fit(data, mask)
        
        #ind = quantize_evecs(ten.evecs, sphere.vertices)
        #eu = EuDX(a=ten.fa, ind=ind, seeds=seedIdx,
        #          odf_vertices=sphere.vertices, a_low=stop_val)
        #tracks = [e for e in eu]
        #return (ten, tracks)
if __name__ == "__main__":
    trk_file = track(dwi_file="/home/nrajamani/ndmg_demo/sub-0025864/ses-1/dwi/sub-0025864_ses-1_dwi.nii.gz",mask_file="/home/nrajamani/ndmg_demo/sub-0025864/ses-1/dwi/sub-0025864_ses-1_dwi_ss_mask.nii.gz",bval="/home/nrajamani/ndmg_demo/sub-0025864/ses-1/dwi/sub-0025864_ses-1_dwi.bval",bvec="/home/nrajamani/ndmg_demo/sub-0025864/ses-1/dwi/sub-0025864_ses-1_dwi.bvec")
