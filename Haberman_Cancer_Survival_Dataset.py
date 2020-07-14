#!/usr/bin/env python
# coding: utf-8

# This is Dataset of Cancer Survival Patient
# Using Haberman_Datasets
# USing Python under ML Project

# In[6]:


#import all necessary modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#Reading the given csv dataset.
haberman=pd.read_csv('/home/harshraj/datasets_haberman.csv')


# In[3]:


# Total Data-points and Features
print(haberman.shape)


# In[5]:


#Total Columns in haberman datasets 
print(haberman.columns)


# In[20]:


#2-D Scatter Plot:
#Always understand the axis:label and scale
sns.set_style("whitegrid");
haberman.plot(kind='scatter',x='Age',y='axil_nodes')


# In[77]:


#Using Seaborn as sns
#Drawing graph according to Surv_status point of view
#According to Age and axil_nodes
#LEgend helps to describe Plot by using scale
sns.set_style("whitegrid");
sns.FacetGrid(haberman,hue="Surv_status",size=4)    .map(plt.scatter,'axil_nodes','Age')   .add_legend();


# # 3D Scatter Plot
# 
# It requires more of mouse operation to interpret data.
# Used to Draw 3D Plot 
# 

# In[60]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman,hue='Surv_status',size=3);
plt.show()


# # observations
# 1.If we take any paricular region in any Hue, we can differentiate with Survival status,
#    i.e, What's the probaibility that they will live or die.
# 2.We can build lines and if-else condition, to build a simple model to classify according to the Survival Status.   
# 3.The Diagonals are the PDF(Probability Density Function) of each feature.

# # HISTOGRAM, CDF,PDF
# Plotting 1D Scatter Plots

# In[67]:


import numpy as np
haberman_Surv_status_Yes=haberman.loc[haberman["Surv_status"]==1]
haberman_Surv_status_No=haberman.loc[haberman["Surv_status"]==2]
plt.plot(haberman_Surv_status_Yes["axil_nodes"],np.zeros_like(haberman_Surv_status_Yes["axil_nodes"]))
plt.plot(haberman_Surv_status_No["axil_nodes"],np.zeros_like(haberman_Surv_status_No['axil_nodes']))
plt.show()


# In[130]:


sns.FacetGrid(haberman,hue='Surv_status',height = 5) .map(sns.distplot,'Op_Year') . add_legend();
plt.show()


# # Observations:
# 
#     Patients with no nodes or 1 node are more likely to survive. There are very few chances of surviving if there are 25 or more nodes.
# 
# # Cumulative Distribution Function(CDF)
# 
# 
# The Cumulative Distribution Function (CDF) is the probability that the variable takes a value less than or equal to x.

# In[116]:


counts1, bin_edges1 = np.histogram(haberman_Surv_status_Yes['axil_nodes'], bins=10, density = True)
pdf1 = counts1/(sum(counts1))
print(pdf1);
print(bin_edges1)
cdf1 = np.cumsum(pdf1)
plt.plot(bin_edges1[1:], pdf1)
plt.plot(bin_edges1[1:], cdf1, label = 'Yes')
plt.xlabel('axil_nodes')
print("***********************************************************")
counts2, bin_edges2 = np.histogram(haberman_Surv_status_No['axil_nodes'], bins=10, density = True)
pdf2 = counts2/(sum(counts2))
print(pdf2);
print(bin_edges2)
cdf2 = np.cumsum(pdf2)
plt.plot(bin_edges2[1:], pdf2)
plt.plot(bin_edges2[1:], cdf2, label = 'No')
plt.xlabel('axil_nodes')
plt.legend()
plt.show()


# # Box Plots and Violin Plots
# 
# The box extends from the lower to upper quartile values of the data, with a line at the median. The whiskers extend from the box to show the range of the data. Outlier points are those past the end of the whiskers.
# 
# Violin plot is the combination of a box plot and probability density function(CDF).

# In[123]:


sns.boxplot(x='Surv_status',y='Age',data=haberman)
plt.show()
sns.boxplot(x='Surv_status',y='Op_Year',data=haberman)
plt.show()
sns.boxplot(x='Surv_status',y='axil_nodes',data=haberman)
plt.show()


# In[126]:


sns.violinplot(x="Surv_status",y="Age",data = haberman,height = 10)
plt.show()
sns.violinplot(x='Surv_status',y='Op_Year',data = haberman,height = 10)
plt.show()
sns.violinplot(x='Surv_status',y='axil_nodes',data = haberman,height = 10)
plt.show()


# # Observations:
# 
#     Patients with more than 1 nodes are not likely to survive. More the number of nodes, lesser the survival chances.
#     
#     A large percentage of patients who survived had 0 nodes. Yet there is a small percentage of patients who had no positive axillary nodes died within 5 years of operation, thus an absence of positive axillary nodes cannot always guarantee survival.
#     
#     There were comparatively more people who got operated in the year 1965 did not survive for more than 5 years.
#     
#     There were comparatively more people in the age group 45 to 65 who did not survive. Patient age alone is not an important parameter in determining the survival of a patient.
#     
#     The box plots and violin plots for age and year parameters give similar results with a substantial overlap of data points. 
#     
#     The overlap in the box plot and the violin plot of nodes is less compared to other features but the overlap still exists and thus it is difficult to set a threshold to classify both classes of patients.

# # Bi-Variate analysis
# 

# # Scatter Plots
# 
# A scatter plot is a two-dimensional data visualization that uses dots to represent the values obtained for two different variables — one plotted along the x-axis and the other plotted along the y-axis.

# In[129]:


sns.set_style('whitegrid')
sns.FacetGrid(haberman, hue = 'Surv_status' , height = 6) .map(plt.scatter,'Age','Op_Year') .add_legend()
plt.show()


# # Observation:
# 
#     Patients with 0 nodes are more likely to survive irrespective of their age.
#     There are hardly any patients who have nodes more than 25.
#     Patients aged more than 50 with nodes more than 10 are less likely to survive.

# # Pair Plots
# 
# By default, this function will create a grid of Axes such that each variable in data will be shared in the y-axis across a single row and in the x-axis across a single column. The diagonal Axes are treated differently, drawing a plot to show the univariate distribution of the data for the variable in that column.

# In[131]:


sns.set_style('whitegrid')
sns.pairplot(haberman, hue='Surv_status', height = 5)
plt.show()


# # Observations:
# 
#     The plot between year and nodes is comparatively better.

# # Multivariate analysis

# # Contour Plot
# 
# A contour line or isoline of a function of two variables is a curve along which the function has a constant value. It is a cross-section of the three-dimensional graph.
# 
# 

# In[133]:


sns.jointplot(x = 'Op_Year', y = 'Age', data = haberman, kind = 'kde')
plt.show()


# # Observation:
# 
#     From 1960 to 1964, more operations done on the patients in the age group 45 to 55.

# # Conclusions:
# 
#     Patient’s age and operation year alone are not deciding factors for his/her survival. Yet, people less than 35 years have more chance of survival.
#     Survival chance is inversely proportional to the number of positive axillary nodes. We also saw that the absence of positive axillary nodes cannot always guarantee survival.
#     The objective of classifying the survival status of a new patient based on the given features is a difficult task as the data is imbalanced.

# In[ ]:




