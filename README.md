Python code for the preprint article "Prediction of Radiologic Outcome-Optimized Dose Plans and Post-Treatment Magnetic Resonance Images: A Proof-of-Concept Study in Breast Cancer Brain Metastases Treated with Stereotactic Radiosurgery" 


Contact: shraddha.pandey@pennmedicine.upenn.edu 
Corresponding:Natarajan.Raghunand@moffitt.org

![Picture13](https://github.com/pandeshraddha/Dose-Map-Prediction/assets/12835584/ec797a6c-a81b-4848-8058-d3faac97e6d5)

![Picture14](https://github.com/pandeshraddha/Dose-Map-Prediction/assets/12835584/32742965-9881-4608-886a-fa350cb25dd5)


The dataset for dose map prediction is available upon request. Please get in touch with the corresponding author for access to the dataset.

The work focuses on directly predicting
the optimum Radiation Therapy (RT) dose maps from the pre-RT mpMRI. It is now well
established that the tumor volume comprises several different microenvironments. Hence,
predicting a voxel-wise dose map from the pre-RT and prescribed/desirable post-RT mpMRI
will yield better control of radionecrosis-related toxicity. Furthermore, it is also important
for the radiation oncologist to simulate voxel-wise radiologic outcomes of specific RT dose
map prescriptions on post-RT mpMRI. To accomplish these two tasks, end-to-end deep
neural networks are trained. The forward model is used to predict post-RT changes on
mpMRI using pre-RT mpMRI when administered with the radiation dose map. A variant
of the pix2pix GAN network is trained to predict post-RT ADC maps, T1wCE, T2w, T1w,
FLAIR MRI from pre-RT mpMRI and the radiation dose maps. The results of the forward
model are validated by identifying the tissue type maps like blood volume, gray matter,
white matter, edema, non-enhancing tumor, contrast enhancing tumor, hemorrhage, fluid
and comparing them with the GT maps. Further, the quantitative validation is carried out
by comparing the percentage of volumes of these tissue type maps from pre-RT, post-RT
and predicted post-RT mpMRI. The results of the forward model are also tested with the
simulated dose maps and comparing the changes on the predicted post-RT ADC maps that
are mechanistically relatable to voxel-level tumor response to therapies. Next, a variant of
pix2pix GAN is trained to predict the radiation dose maps from the pre-RT ADC maps and
the prescribed post-RT ADC maps. This is called as the inverse model. It is determined
from the simulated results that to achieve higher ADC values, higher RT dose maps are
required.
