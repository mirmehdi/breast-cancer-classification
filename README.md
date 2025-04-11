# CASE STUDY: BREAST CANCER CLASSIFICATION
# Dr. Ryan Ahmed


- Predicting if the cancer diagnosis is benign or malignant based on several observations/features 
- 30 features are used, examples:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry 
        - fractal dimension ("coastline approximation" - 1)

- Datasets are linearly separable using all 30 input features
- Number of Instances: 569
- Class Distribution: 212 Malignant, 357 Benign
- Target class:
         - Malignant
         - Benign


https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

![image.png](attachment:image.png)


## Methods: 
- This code compares all machine learning algorithms in their defauld params to see which one is appropriate for this classification. 
- Then based on cross_validation and f1 score decides which method is better. 
- As final step we use Grid_search to find the best parameters. 
- A summery report of statisitcal analysis & results are  saved in "breast_cancer_descriptive_analysis.txt"



