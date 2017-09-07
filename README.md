# opencv-haralickfeatures
Implementation of GLCM Haralick Features using openCV, based in Haralick et. al (1979)

Calculates the Co-ocurrence matrix from an openCV \<uchar\> Mat object and process it to extract relevant texture information according to the paper

12 from 14 features of the paper are implemented. "Maximal Correlation Coefficient" was ignored for being notoriously unstable in the literature and "Sum of Squares: Variance" was avoided for ambiguity of the meaning of μ in the formulae

# References
Textural Features for Image Classification, Haralick et. al (1979), available in: http://haralick.org/journals/TexturalFeatures.pdf

Fast Calculation of Haralick Texture Features, Eizan Miyamoto and Thomas Merryman Jr (2008) for clarification of meaning of some features. Available in:
https://www.inf.ethz.ch/personal/markusp/teaching/18-799B-CMU-spring05/material/eizan-tad.pdf
