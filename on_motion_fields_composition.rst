#Composition of two motion fields

The composition of two motion fields essentially combines the transformations of the 
individual motion fields to create a new motion field that represents their combined effect.

## definition:

The composition of two motion fields F and G can be defined as follows:

H(x) = F(G(x))

where H is the resulting motion field, F is the motion field obtained from the registration of
'im_0_i' with 'im_t_j', and G is the motion field obtained from the registration of 'im_1_i' 
with 'im_t_j'. (see the study for notations)
 
## steps to follow:

Therfore, we can follow these steps to compute the composition of the motion fields:

1- Given motion fields F (from 'im_0_i' registration) and G (from 'im_1_i' registration), we 
need to create a grid for each motion field that represents the pixel-wise transformations.
 
2- Apply the motion fields F and G to the grids of 'im_0_i' and 'im_1_i' respectively, 
using the warp function (see the provided utils.py file). This will create two new images 
'warped_im_0_i' and 'warped_im_1_i', which represent the transformed versions of 'im_0_i' and 
'im_1_i' according to the respective motion fields.

3- Now, the composition motion field H is equivalent to the transformation that takes 'im_0_i' 
to 'warped_im_1_i'. Therefore, we can compute the composition motion field H by registering 
'im_0_i' with 'warped_im_1_i' using the CR network. 
This will give us the combined motion field H that represents the alignment between 'im_0_i'
 and 'im_1_i'.

By following these steps, we can compute the composition of the two motion fields and ensure 
consistency between the registrations of 'im_0_i' and 'im_1_i' with 'im_t_j'. 
The resulting motion field H should enable alignment between the two images 'im_0_i' and 'im_1_i'.
