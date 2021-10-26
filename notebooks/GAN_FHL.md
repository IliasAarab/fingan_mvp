---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

Here's a simple footnote,[^1] and here's a longer one.[^bignote]

[^1]: This is the first **footnote**.

[^bignote]: Here's one with multiple paragraphs and code.

    Indent paragraphs to include them in the footnote.

    `{ my code }`

    Add as many paragraphs as **you** like.

<!-- #region colab_type="text" id="O0aCFQLqTAkc" -->
# Stress Testing with GANs v2.4: Generating FHL based data

*The current notebook implements our revised GAN model (WGAN-GP-Mixed) to generate data based on the FHL public dataset. The improved Generative Adversarial Network is able to work with highly structured multivariate distributions containing both continuous and discrete marginal distributions. The new model is able to overcome any gradient vanishing problems we encountered before and is based on the set of papers described below. The Federal Home Loan Bank Purchased Mortgage dataset is obtained through the [Federal Housing Finance Agency
](https://www.fhfa.gov/DataTools/Downloads/Pages/Public-Use-Databases.aspx) and contains granular loan-level information of US-based mortgages. The dataset contains 82 variables, both continuous and discrete, and has some variables that exhibit a mixed distribution with a Dirac zero delta point. The dataset contains about 65.000 instances making it easily to digest in test settings. The high degree of variability in the data makes it especially challenging to learn the underlying manifold for our GAN model.* 

----

[link](#Importing-libraries-and-dataset)

Setup:
  
- [Step 1:](#Environment-variables) Setup working environment
  
- [Step 2:](#The-Federal-Housing-Finance-Agency-dataset) The FHL dataset
  
- [Step 3:](#Preprocessing-of-data) Preproccesing of data
  
- [Step 3:](#Setting-up-our-GAN-model) Implement a composite GAN NN
  
- [Step 4:](#Training-of-the-compiled-GAN-model) Training of the compiled GAN model
  
- [Step 5:](#Evaluating-the-performance-of-our-trained-model) Evaluating the performance of our trained model

- [Step 6:](#Evaluate-the-similarity-between-real-and-artificial-distributions) Evaluate the *similarity* between real and artificial distributions
  
- [Step 7:](#Save-the-calibrated-Generator-and-the-synthetic-dataset) Save the calibrated Generator and the synthetic dataset

---

Main papers utilized:
  
- Overview of GANs: [GAN Google](https://developers.google.com/machine-learning/gan/summary)
  
- Seminal GAN paper of Goodfellow et al.: [GAN Seminal paper](https://arxiv.org/abs/1406.2661)
  
- Instability problems with GANs: [Stabilizing GANs](https://arxiv.org/pdf/1910.00927.pdf)
  
- Wasserstein GAN to deal with vanishing gradients: [WGAN](https://arxiv.org/abs/1701.07875) 
  
- Improved WGAN with Gradient Penalty: [WGAN-GP](https://arxiv.org/abs/1704.00028)
  
- How to add custom gradient penalty in Keras: [WGAN and Keras](https://keras.io/examples/generative/wgan_gp/) and [Change model.fit()](https://keras.io/guides/customizing_what_happens_in_fit/)
  
- Multi-categorical variables and GAN: [Multi-categorical GAN](https://arxiv.org/pdf/1807.01202.pdf)
  
- Alternative to WGAN: [DRAGAN](https://arxiv.org/abs/1705.07215)
  
- Conditional DRAGAN on American Express dataset: [DRAGAN & American Express](https://arxiv.org/pdf/2002.02271.pdf)
  
- Mulitvariate KL-divergence by KNN: [KL by KNN](https://ieeexplore.ieee.org/document/4035959)
  
- Multivariate KL-divergence by  KNN p2: [KL by KNN](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.422.5121&rep=rep1&type=pdf)
  
<!-- #endregion -->

<!-- #region colab_type="text" id="Mb17kI2yTAkd" -->

## Environment variables


Connect to Google Colab Directory containing our repository
(only executed on GColab)
<!-- #endregion -->

```python
# Initiate notebook (only to be ran once, or when kernel restarts)
activated_reload= False
activated_chdir= False
GCOLAB= False
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 30647, "status": "ok", "timestamp": 1600802580470, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="WG2gtnD6TAke" outputId="34bc0955-3570-455a-a3d3-c1f8b454b6de"
if GCOLAB:
    from google.colab import drive
    import sys
    drive.mount('/content/gdrive/')
    sys.path.append('/content/gdrive/My Drive/Stress Testing with GANs/notebooks')
```

## Settings & environment setup 

```python
# Settings for auto-completion: In case greedy is turned off and jedi blocks auto-completion
%config IPCompleter.greedy=True
%config Completer.use_jedi = False

# Autoreload modules: in case in-house libraries aren't finalised
if not activated_reload:
    activated_reload= True
    %load_ext autoreload
    %autoreload 2
    

# Set working directory for user to main project directory 
import os 
if not activated_chdir:
    activated_chdir= True
    os.chdir('..')
    print('Working directory: ', os.getcwd())
```

<!-- #region colab_type="text" id="RxjuYiyWTAkp" -->
## Import all the functions of the source code (including libraries)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 71} colab_type="code" executionInfo={"elapsed": 3766, "status": "ok", "timestamp": 1600802587801, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="fF0HDGDQTAkq" outputId="a069be7d-fb5c-4558-d2e5-2a778d8fd899"
# TODO: modularize source code
from src.GAN_functions import *

# Import specific libraries
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
```

```python
## Libraries settings

# matplotlib
%config InlineBackend.figure_format = 'png'
%matplotlib inline
import matplotlib.font_manager
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = "serif"
#plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = r"\usepackage{subdepth}, \usepackage{type1cm}"

# plotly
pio.renderers.default = "notebook+pdf"  

# numpy
np.set_printoptions(precision=2)

# pandas
pd.set_option('display.notebook_repr_html', True)
def _repr_latex_(self):
    return "\\begin{center} \n %s \n \\end{center}" % self.to_latex()
pd.DataFrame._repr_latex_ = _repr_latex_  # monkey patch pandas DataFrame
```

# The Federal Housing Finance Agency dataset

<!-- #region colab_type="text" id="26zzizpQTAkz" -->
## Importing the FHL dataset 

We have three general ways of importing the data: 

1. Use a local environment and extract the data from the same environment
2. Use a cloud environment and extract data from the local environment 
3. Use a cloud environment and extract data from the same cloud environment
<!-- #endregion -->

<!-- #region colab_type="text" id="A4VUD01rTAk0" -->
### Jupyter notebook on local environment 
<!-- #endregion -->

```python colab={} colab_type="code" id="gD2tAPIPTAk1"
if not GCOLAB:
    filename= 'FHLbank.csv'
    path= os.path.join('data/raw', filename)
    FHL_bank = pd.read_csv(path)
```

<!-- #region colab_type="text" id="k1nNHYYbTAlC" -->
### Google Colab when importing dataset from local environment  
<!-- #endregion -->

```python colab={} colab_type="code" id="2N_Pq35cTAlD"
if GCOLAB:
    # Import files module 
    from google.colab import files
    # Upload PPNR dataset into GColab 
    ppnr = files.upload()
    # Read the file
    import io
    ppnr = pd.read_csv(io.BytesIO(ppnr['FHLbank.csv']))
    # Convert to the right data type 
    ppnr = ppnr.astype(dtype = 'float64') 
```

<!-- #region colab_type="text" id="pKZC_jK_TAlI" -->
### Google Colab when importing dataset directly from Google Drive

<!-- #endregion -->

```python colab={} colab_type="code" executionInfo={"elapsed": 21686, "status": "ok", "timestamp": 1600802615141, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="WSGb76ULTAlK"
if GCOLAB:
    # Read csv file into Colaboratory
    !pip install -U -q PyDrive
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials

    # Authenticate and create the PyDrive client
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
```

```python colab={} colab_type="code" executionInfo={"elapsed": 3821, "status": "ok", "timestamp": 1600802622106, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="1BaBrKXJTAlO"
if GCOLAB:
    # Shareable link from the dataset in Google drive 
    link= 'https://drive.google.com/file/d/1lqeFxTx3MEZKhwowRWzUKTdYPpwvWDUd/view?usp=sharing'
    #we only need the id-key portion of the link
    id= '1lqeFxTx3MEZKhwowRWzUKTdYPpwvWDUd'

    # Import dataset
    downloaded = drive.CreateFile({'id':id}) 
    downloaded.GetContentFile('FHLbank.csv')  
    FHL_bank = pd.read_csv('FHLbank.csv')
```

## Quick overview of the dataset

The dataset contains 65,703 mortgages derived in 2018, with no missing
values. The median borrower(s) total annual income amounts to
$\$95,000$ USD, whereas the median family income of the
immediate area where the house is purchased is only
$\$73,600$ USD. The median interest rate is set to 4.63\%
whereas the average rate is slightly lower (4.55\%) indicating that the
interest rate is slightly left-skewed. When looking at the amount
borrowed, we see that the median amount is set to
$\$211,000$ USD and an average amount of
$\$237,000$ USD, indicating a right-skewed distribution.

```python
# Missing values 
print('Missing values:', FHL_bank.isnull().values.any())
# Quick overview
display(FHL_bank.sample(n=12,axis='columns').head())
```

```python
cols= ['Year', 'Income', 'CurAreY', 'LTV', 'Rate', 'Amount']
FHL_bank[cols].describe()
```

```python
fig = make_subplots(rows=1, cols=3, )
cols = ['Rate', 'Amount', 'LTV']
for i, eachCol in enumerate(cols):
    fig.add_trace(go.Histogram(
        x=FHL_bank[eachCol], name=eachCol, ), row=1, col=i+1)
    fig['layout'][f'xaxis{i+1}']['title']= eachCol
fig.update_layout(
    title_text='Histogram of Interest rate, Amount Borrowed and the LTV')

config = {'staticPlot': True}
fig.show(config=config)
```

<!-- #region -->
## Quick overview of the dataset: part 2


When looking at discrete distributions, we see that multiple marginal distributions are highly imbalanced. The purpose of the borrower is in more than 99% of the cases either to purchase a new home or to refinance a current home, only .02% of borrowers enter a mortgage for a new construction. The imbalance is seen when looking at the number of borrower's for one mortgage, with more than 99% of mortgages having either one or two underlying borrowers, and only a small percentile of mortgages (less than 1%) consist of three or four borrower's. The majorities of borrower's are identified as white (83%), about 10% of borrower's didn't provide their race information, and the remaining 7% is either Native, Asian, African American or Hawaiian.


Two interesting characteristics can be noted when looking at the age distribution of the data. First note the slightly bimodality of the distribution, with the first being located around the 35 year age and the second one at the 55 year mark. Secondly, note the large amount of mass concentrated at the right side of the distribution. These borrower's haven't included their age in the application form resulting in the value *99* being entered in the dataset. We can thus see the age distribution as a mixed distribution containing on the one hand a bimodal Gaussian mixture distribution and the other hand a Dirac delta point with all mass concentrated at *99*. The last point is highly relevant, as there might a consistent reasoning as to why applicants choose not to disclose their age and their performance on the repayment of the mortgage. When estimating a Gaussian mixture model we do see indeed a composition in three unimodel Gaussians with means of respectively 54.9 years, 34 years and 99 "years". It is interesting to see how and if our GAN model is going to be able to capture these intricities and generate them successfully.




<!-- #endregion -->

```python
# Compute relative frequencies
cols= ['Purpose', 'NumBor', 'BoRace', 'BoGender']
rel_freq= FHL_bank[cols].apply(pd.Series.value_counts, args=(True,))
rel_freq.index= [f'class {i}' for i in range(len(rel_freq))]
rel_freq
```

```python
cols= ['Purpose', 'NumBor', 'BoRace', 'BoGender', ]

fig = make_subplots(rows= len(cols), cols=1, )
for i, eachCol in enumerate(cols):
    fig.add_trace( go.Bar( x= FHL_bank[eachCol].value_counts(), name= eachCol), row= i+1, col=1)
    fig['layout'][f'yaxis{i+1}']['title']= eachCol

fig.update_layout(title_text= 'Discrete sample distributions')
config = {'staticPlot': True}
fig.show(config= config)
```

```python
cols= ['BoAge']
fig= go.Figure(data=[go.Histogram(x= FHL_bank['BoAge'])])
fig.update_layout(title_text= "Histogram of borrower's age")
fig.add_annotation(
        x=99,
        y=5000,
        xanchor= 'right',
        xref="x",
        yref="y",
        text="Shifted Dirac delta point",
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        align="left",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=-20,
        ay=-5,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8
        )

fig.add_annotation(
        x=40,
        y=3000,
        xanchor= 'center',
        xref="x",
        yref="y",
        text="Bimodal distribution",
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
            ),
        #align="left",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=0,
        ay=-50,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8
        )



config = {'staticPlot': True}
fig.show(config= config)
```

```python
# Estimate a Gaussian mixture model on the Age distribution
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(3).fit(FHL_bank[cols])
# Get the estimated parameters
gm_params = pd.DataFrame(
    np.concatenate(
        (gm.means_, gm.covariances_.reshape(gm.means_.shape)), axis=1),
    columns=['Means', 'Variance'],
    index=['firstDecomposition', 'secondDecomposition', 'thirdDecomposition'])
print(f"The estimated Gaussian mixture model has the following parameters:")
gm_params
```

<!-- #region colab_type="text" id="XLEScavqTAlW" -->

# Preprocessing of data 

We first need to separate the data into continous variables and discrete variables as our neural network digest these type of variables in a distinct different way. We also remove variables that don't contain any real information like IDs, these can easily be later on added and anonymized. 
<!-- #endregion -->

We first need to separate the data into continous variables and discrete variables as our neural network digest these type of variables in a distinct different way. We also remove variables that don't contain any real information like IDs, these can easily be later on added and anonymized. $d$

```python colab={} colab_type="code" executionInfo={"elapsed": 673, "status": "ok", "timestamp": 1600802626816, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="oWC9HppNTAlW"
# Store column names 
redudantColumns=['Year', 'AssignedID', 'FeatureID', 'Rent1', 'Rent2', 'Rent3', 'Rent4', 'RentUt1', 'RentUt2', 'RentUt3',
                'RentUt4', 'ArmIndex', 'ArmMarg', 'PrepayP', 'Bed1', 'Bed2', 'Bed3', 'Bed4',
                'Aff1', 'Aff2', 'Aff3', 'Aff4', 'ArmIndex', 'ArmMarg', 'PrepayP', 'FedFinStbltyPlan', 'GSEREO',
                'AcquDate', 'Coop', 'Product','SellType', 'HOEPA','LienStatus']

continuousColumns= ['MSA', 'Tract', 'MinPer', 'TraMedY', 'LocMedY', 'Tractrat', 'Income', 'CurAreY', 'IncRat',
                   'UPB', 'LTV', 'BoAge', 'CoAge', 'Rate','Amount','Front','Back']

nominalColumns= ['Bank', 'FIPSStateCode', 'FIPSCountyCode', 'MortDate',  'Purpose', 'FedGuar', 'Term', 'AmorTerm', 
                'NumBor', 'First','CICA','BoRace','CoRace','BoGender',
                 'CoGender', 'Geog', 'BoCreditScore', 'CoBoCreditScore', 'PMI', 'Self', 'PropType', 
                'BoEth','Race2','Race3','Race4','Race5','CoEth','Corace2','Corace3','Corace4','Corace5',
                 'SpcHsgGoals','AcqTyp']


# Nominal data 
datasetNominal= FHL_bank[nominalColumns]

# Continuous data (including ordinal data)
datasetContinuous= FHL_bank[continuousColumns]

# Get number of categorical values of each nominal variable 
nominalColumnsValues= datasetNominal.nunique().values
```

<!-- #region colab_type="text" id="PM_RhTkjTAla" -->
## Preprocessing of nominal data 

Nominal (discrete) data creates the additional problem that we end up with a non-differentiable bottleneck in our neural network leading to problems down the road. To deal with this issue, we follow the solutions implemented in NLP based networks in asimplified manner as the number of categories of our nominal is much smaller compared to NLP vocabularies. 

We use the following setup: 

1. we transform discrete marginals into a onehot encoding as:

\begin{equation}
{\bf n_{i,j} } \leftarrow n_{i,j}
\end{equation}


with $n_{i,j}$ the i-th sample with the jth discrete variable and ${\bf n_{i,j} }$ its one-hot encoded vector representation

2. We add a small amount of noise to the one-hot vector: 

\begin{equation}
{\bf n_{i,j}^{noise} } \leftarrow n_{i,j} + 
\sqrt{\sigma}\epsilon
\end{equation}

with $\epsilon$ either a standard Gaussian distribution or uniform distribution and $\sigma$ either the variance of the Gaussian distribution or a scaling constant of the uniform distribution

3. Normalize the noisy one-hot vector in order to have a coherent probability measure:

\begin{equation}
{\bf n_{i,j}^{noise} } \leftarrow  
{\bf n_{i,j}^{noise} } / \sum_{k=1}^{J}{\bf n_{i,j}^{noise, k} }
\end{equation}



<!-- #endregion -->

```python colab={} colab_type="code" executionInfo={"elapsed": 1593, "status": "ok", "timestamp": 1600802629117, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="z2L79kwPTAlb"
nameFeatures = nominalColumns
ohe = OneHotEncoder(sparse= False) #create encoder object 
datasetEncoded = ohe.fit_transform(datasetNominal) #transform nominal data 
datasetEncoded = pd.DataFrame(datasetEncoded) #transform output to df 
datasetEncoded.columns = ohe.get_feature_names(nameFeatures) #name columns appropriately 
```

```python colab={} colab_type="code" executionInfo={"elapsed": 981, "status": "ok", "timestamp": 1600802629394, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="tJ41wJACTAlf"
# Transforms ohe dataset to list of individual ohe variables 
def variableSeparator(nominalColumnValues, nominalDataset):
    '''
    # nominalColumnValues: A list containing the number of unique values of each variable 
    # nominalDataset: a dataset containing one-hot encoded nominal variables 
    Separates all of the individual variables into a list, so it's easy to perform computations on individual variables 
    '''
    dim= len(nominalColumnsValues)
    variablesSeparated= []
    for eachVariable in range(dim):
        if eachVariable == 0:
            tmp= nominalDataset[:, eachVariable:eachVariable+nominalColumnsValues[eachVariable]]
            variablesSeparated.append(tmp)
            idx= eachVariable+nominalColumnsValues[eachVariable]
        else: 
            tmp= nominalDataset[:, idx:idx+nominalColumnsValues[eachVariable]]
            variablesSeparated.append(tmp)
            idx+= nominalColumnsValues[eachVariable]
            
    return variablesSeparated
```

```python colab={} colab_type="code" executionInfo={"elapsed": 3309, "status": "ok", "timestamp": 1600802632249, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="gClKh9DjTAlk"
# Create noise 
noise = np.random.uniform(0, 0.2, datasetEncoded.shape) #noise level of 0.2 is the standard 
# Add noise to dataset 
dataset_with_noise = datasetEncoded.values + noise 

# Normalize each variable separately (from ohe to probabilities)
datasetNominalNormalized = np.array([]).reshape(len(dataset_with_noise), -1) #initialize 
variablesSeparated= variableSeparator(nominalColumnValues = nominalColumnsValues, nominalDataset= dataset_with_noise)
for eachVariable in variablesSeparated: 
    tmp= ( eachVariable / np.sum(eachVariable, axis= 1)[:, None] )
    datasetNominalNormalized= np.concatenate( (datasetNominalNormalized, tmp), axis= 1)
```

<!-- #region colab_type="text" id="89xBT-KyTAln" -->
## Preprocessing of continous data 

Most papers recommend using a bounded activation function to train a GAN model as it leads to more stable results. As mentioned in Radford et al., Unsupervised representation learning with DCGAN  “The ReLU activation (Nair & Hinton, 2010) is used in the generator with the exception of the output layer which uses the Tanh function. We observed that using a bounded activation allowed the model to learn more quickly to saturate and cover the color space of the training distribution.” 
Thus we use the Tanh activation function when generating continuous data with the Generator.  To make the output of the Generator match with the input dataset to the Discriminator we preprocess the continuous data to [-1,1]. 
However, it might be the case that other bounded activation functions yield better results and thus this should be regarded as  a hyperparameter that can be calibrated

<!-- #endregion -->


```python colab={} colab_type="code" executionInfo={"elapsed": 1818, "status": "ok", "timestamp": 1600802632255, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="nwRK7y0lTAlo"
maximum = np.max(datasetContinuous)
minimum = np.min(datasetContinuous)
datasetContinuousNormalized= ( 2 * (datasetContinuous - minimum) / (maximum - minimum) - 1 )
```

<!-- #region colab_type="text" id="1YgGhXTVTAlr" -->
## Create final processed dataset
<!-- #endregion -->

```python colab={} colab_type="code" executionInfo={"elapsed": 1088, "status": "ok", "timestamp": 1600802633542, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="iD9xArNRTAlr"
dataset= np.concatenate((datasetContinuousNormalized, datasetNominalNormalized), axis=1) 
```

<!-- #region -->


# Setting up our GAN model
We can either generate our data from the modified Vanilla GAN or our newest version of the WGAN-GP-MIXED model. 
<!-- #endregion -->

```python
# Turn the desired model on:
VGAN= False
WGAN_GP_MIXED= True
```

## Modified Vanilla GAN model 

```python colab={} colab_type="code" id="1u9vYGU-TAlx"
if VGAN:
    ## DISCRIMINATOR

    # Shape of the dataset 
    input_shape= dataset.shape[1]
    # Create the discriminator
    discriminator = discriminator_setup_CTGAN(input_shape)

    ## GENERATOR 

    # Shape of the latent space of the generator 
    latent_dim= 100 #used in many papers 
    # create the generator
    generator = generator_setup_CTGAN(input_shape, latent_dim,)

    ## GAN

    # create the gan
    gan_model = gan_setup(generator, discriminator)
```

<!-- #region colab_type="text" id="OifzNpr9TAl0" -->
## WGAN-GP-Mixed
- TODO: Modify to easier compile the model (WIP)

The basic GAN suffers from a vanishing gradient of the Discriminator which results in the Discriminator failing to provide useful gradient information (during backpropagation) to the Generator as soon as the Discriminator is well-trained. In contrast, the WGAN-GP Discriminator does not saturate and converges to a linear function with clean gradients. 
An easy way to see this is to run both the basic GAN and WGAN-GP on the FHL dataset and evaluate the training process. 
A more detailed discussion can be found in Arjovsky et al., Wasserstein GAN. 

<!-- #endregion -->

```python colab={} colab_type="code" executionInfo={"elapsed": 1084, "status": "ok", "timestamp": 1600802670110, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="GmlbkYbtTAl0"
if WGAN_GP_MIXED:
    ############################### INITIALIZE PARAMETERS #########################



    # Optimizer for both the networks
    #learning_rate=0.0002, beta_1=0.5 are recommened
    generator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9)



    # Define the loss functions to be used for discrimiator
    # This should be (fake_loss - real_loss)
    # We will add the gradient penalty later to this loss function
    def discriminator_loss(Xreal, Xfake):
        real_loss = tf.reduce_mean(Xreal)
        fake_loss = tf.reduce_mean(Xfake)
        return fake_loss - real_loss


    # Define the loss functions to be used for generator
    def generator_loss(Xfake):
        return -tf.reduce_mean(Xfake)


    # Epochs to train
    epochs = 1000
    noise_dim= 512
    batch_size= 512

    # MODEL 
    d_model = discriminator_setup_WGANGP(dataset.shape[1])

    g_model = generator_setup_WGANGP_GENERIC(dataset.shape[1],noise_dim, 256, 2, 
                                             nominalColumnsValues, nominalColumns, datasetContinuousNormalized.shape[1])
    # Get the wgan model
    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=noise_dim,
        discriminator_extra_steps=5,
    )

    # Compile the wgan model
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )


    # Custom callback to get KL-divergence 
    cbk= GANMonitor(dataset, 1000, noise_dim, 10)
```

<!-- #region colab_type="text" id="EDpe6qlGTAl3" -->


# Training of the compiled GAN model

We can either train one of our compiled models (VGAN or WGAN-GP-Mixed) or load in a previously trained model
<!-- #endregion -->

```python
# Choose to train a model or load a previous trained model  (locally or from the cloud)
TRAIN_NEW= False
LOAD_LCL= True
LOAD_CL= False
```

<!-- #region colab_type="text" id="NvHUADkPTAl4" -->
## Vanilla GAN
<!-- #endregion -->

```python colab={} colab_type="code" id="O8dUxtgCTAl4"
if VGAN and TRAIN_NEW:
    ## Train model
    n_epochs= 10
    batch_size= len(dataset)
    train_gan(discriminator, generator, gan_model, dataset, latent_dim, n_epochs, batch_size,True,
         'desktop') #saves generator after every epoch 
```

<!-- #region colab_type="text" id="bjH-WzPVTAl6" -->
## WGAN-GP-Mixed
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" executionInfo={"elapsed": 6265740, "status": "ok", "timestamp": 1600808938715, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="B2ql39jmTAl7" outputId="3fe40f5a-c6cb-4934-be4d-2f4de15fe5e1"
if WGAN_GP_MIXED and TRAIN_NEW:
    ## Train model
    history = wgan.fit(dataset, batch_size= batch_size, epochs= epochs, callbacks= [cbk])

    ## Save Generator
    !mkdir -p saved_model
    g_model.save('/content/gdrive/My Drive/Stress Testing with GANs/generator_model')
```

<!-- #region colab_type="text" id="qmch9d8YTAl9" -->
## Loading of a previous trained model

<!-- #endregion -->

<!-- #region colab_type="text" id="KK5DjAiJTAl-" -->
### Load model saved in local environment 
<!-- #endregion -->

```python colab={} colab_type="code" id="TdUovXEXTAl-"
if LOAD_LCL:
    mdl= 'mdl/generator_model'
    g_model = tf.keras.models.load_model(mdl)
```

<!-- #region colab_type="text" id="AbEXpc0cTAmB" -->
### Load model saved in Google Drive 
<!-- #endregion -->

```python colab={} colab_type="code" id="drPhMZTlTAmC"
if LOAD_CL:
    mdl= r'/content/gdrive/My Drive/Stress Testing with GANs/generator_model'
    g_model = tf.keras.models.load_model(mdl) 
```

<!-- #region colab_type="text" id="RJL0O9KETAmK" -->



# Evaluating the performance of our trained model

<!-- #endregion -->

<!-- #region colab_type="text" id="ZzFmntz1TAmL" -->
## Loss functions for WGAN-GP

- Still a work-in-progress
 
We have the following minimax problem with the WGAN-GP model (ignoring the GP for now): 

\begin{equation}
\min_{G}\max_{D}
 \mathbb{E_{x\sim\text{p(X)}}}\{D(x)\} - 
 \mathbb{E_{z\sim\text{p(Z)}}}\{D(G(z))\}
\end{equation}
  
with
$p(X)$ the true distribution, $p(Z)$ the latent distribution, $G(z)$ the Generator and $D(x)$ the Discriminator. 
  
Which results in the following loss functions: 
  
Discriminator: 
\begin{equation}
\min_{D} \mathbb{E_{z\sim\text{p(Z)}}}\{D(G(z))\} -
\mathbb{E_{x\sim\text{p(X)}}}\{D(x)\}
\end{equation}
  
Generator:
\begin{equation}
\min_{G} - \mathbb{E_{z\sim\text{p(Z)}}}\{D(G(z))\}
\end{equation}

We approximate the Expected values by taking the arithmetic means. 
  
Since the Discriminator has an output space of $[-\infty, +\infty]$ we can expect quite erratic behaviour which would result in problems during the backpropagation phase. To ensure a well-behaved function we constrain the Discriminator to be 1-Lipschitz continuous. We can do this by either weightclipping (WGAN) or adding a gradient penalty (WGAN-GP). Weightclipping is suboptimal so we use the Gradient Penalty. 
  
The Gradient Penalty ensures that the norm of the gradient of the Discriminator (with the input being an interpolation of the real and fake distribution) is at most one, which ensures that the Discriminator is 1-Lipschitz continuous (see Proof of Proposition 1, page 12, Improved WGANs). The Discriminator loss function then becomes: 
  
\begin{equation}
\min_{D} \mathbb{E_{z\sim\text{p(Z)}}}\{D(G(z))\} -
\mathbb{E_{x\sim\text{p(X)}}}\{D(x)\} + 
\lambda\mathbb{E_{w\sim\text{p(W)}}}\{(\lVert  \nabla D(w)  \rVert_{2} - 1)^2\}
\end{equation}
  
with $\lambda$ a Langrage multiplier and $p(W)$ an interpolated distribution. 
  
During backpropagation the weights of the Discriminator will be adjusted to minimize the added Gradient Penalty, ensuring that the weights and Discriminator conform to the 1-Lipschitz requirement in a more natural way than directly clipping the weights.
  
Within the current notebooks (07/09) the two loss functions are defined within the current notebook (see above), whereas the Gradient Penalty is defined within the src code within the WGAN class by utilizing Tensorflow to create interpolated distributions. The loss function of the Discriminator and the Gradient Penalty are then added together during the training (when calling the fit() function). 
<!-- #endregion -->

## How should I interpret the loss functions of WGAN-GP?

The loss function of the discriminator is an approximation of the Earth-Mover distance (i.e. the minimum transportation cost when transporting mass of probability Q in order to approximate probability P) up to a constant scaling factor dependent on the Discriminator’s architecture and the Gradient Penalty. The constant scaling factor makes it hard to compare the loss functions between different model specifications. Note also, that the loss function is simply an approximation of the true EM-distance, so it’s hard to know how closely the loss function is to the true EM-distance. 
  
However,  we can use the loss function as an indication of how well the training relatively improves the performance of our model.  A lower loss would indicate an improvement in the generated distribution. In our case the loss function is inverted so an increase in the loss would indicate an improvement in the generated distribution. Examples of plots of the loss function can be found back in the  Wasserstein GAN and Improved Training of WGANs papers. 
  
The Generator is dependent on the Discriminator during the backpropagation phase and thus the loss function of the Generator does not provide a meaningful absolute metric. However, we should expect the generator to oscillate between a certain range during convergence or to gradually increase/decrease, without any erratic behaviour. 
  
The KL-divergence quantifies how much one probability distribution differs from another probability distribution. The range of the KL-divergence is $[0, +\infty]$ with a value of zero indicating that the two distributions are identical. Thus we should expect the KL-divergence to decrease during the training phase and converge towards zero. 



```python colab={"base_uri": "https://localhost:8080/", "height": 350} colab_type="code" executionInfo={"elapsed": 1887, "status": "ok", "timestamp": 1600809496048, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="dG8l1kI343Bq" outputId="d35db190-71ea-450c-ed31-074b9458024f"
# Plots the loss functions of Discriminator and Generator and the KL-divergence obtained during training 
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

## Main function

def plot_metric(history, cbk):
  g_loss= history.history['g_loss']
  d_loss= history.history['d_loss']
  kl= np.array(cbk.kl_tracker)

  #function to set our spines invisible 
  def make_patch_spines_invisible(ax):
      ax.set_frame_on(True)
      ax.patch.set_visible(False)
      for sp in ax.spines.values():
          sp.set_visible(False)



  # ------------------        
  ## Main Graph 
  # ------------------


  #setup figure 
  fig, host = plt.subplots(figsize=[20, 5])
  fig.subplots_adjust(right=0.75)
  par1 = host.twinx()
  par2 = host.twinx()
  # Offset the right spine of par2.  The ticks and label have already been
  # placed on the right by twinx above.
  par2.spines["right"].set_position(("axes", 1.1))
  # Having been created by twinx, par2 has its frame off, so the line of its
  # detached spine is invisible.  First, activate the frame but make the patch
  # and spines invisible.
  make_patch_spines_invisible(par2)
  # Second, show the right spine.
  par2.spines["right"].set_visible(True)
  # Ready to plot our graphs 
  epochs = range(1, len(g_loss) + 1)
  p1, = host.plot(epochs, kl, 'b', label="KL-Divergence")
  p2, = par1.plot(d_loss, 'r', label="d_loss")
  p3, = par2.plot(g_loss, 'g', label="g_loss")
  # Name labels appropriately 
  host.set_xlabel("Epochs", fontweight='bold')
  host.set_ylabel("KL-Divergence", fontweight='bold')
  par1.set_ylabel("d_loss", fontweight='bold')
  par2.set_ylabel("g_loss", fontweight='bold')
  # Keep coloring uniform across the entire figure 
  host.yaxis.label.set_color(p1.get_color())
  par1.yaxis.label.set_color(p2.get_color())
  par2.yaxis.label.set_color(p3.get_color())
  # Clean up ticks 
  tkw = dict(size=4, width=1.5)
  host.tick_params(axis='y', colors=p1.get_color(), **tkw)
  par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
  par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
  host.tick_params(axis='x', **tkw)
  # Add outside legend and give title
  lines = [p1, p2, p3]
  host.legend(lines, [l.get_label() for l in lines], title= 'LEGEND', bbox_to_anchor=(1.30, 1),)
  plt.title('Generator/Discriminator Loss functions and the KL-divergence', fontweight='bold')


  # ------------------------------
  ## Zoom-ins to get more detail 
  # ------------------------------

  # Zoom into a part of KL-divergence graph
  axins = zoomed_inset_axes(host, 2, loc= 5) # (graph, zoom-in factor, location on graph to display the zoom)
  #What to zoom in on
  axins.plot(epochs, kl)
  x1, x2, y1, y2 = len(kl) - 100, len(kl), 0, 5 # specify the limits
  axins.set_xlim(x1, x2) # apply the x-limits
  axins.set_ylim(y1, y2) # apply the y-limits
  #Add connecting lines to original plot 
  mark_inset(host, axins, loc1=3, loc2=4, fc="none", ec="0", )

  #Zoom into a part of the graph
  axins = zoomed_inset_axes(par1, 2, loc= 10) # (graph, zoom-in factor, location on graph to display the zoom)
  #What to zoom in on
  axins.plot(epochs, d_loss, 'r')
  x1, x2, y1, y2 = len(d_loss) - 100, len(d_loss), -0.1, 0 # specify the limits
  axins.set_xlim(x1, x2) # apply the x-limits
  axins.set_ylim(y1, y2) # apply the y-limits
  #Add connecting lines to original plot 
  mark_inset(par1, axins, loc1=2, loc2=4, fc="None", ec="0", )

  
  
plot_metric(history=history, cbk= cbk)
```

<!-- #region colab_type="text" id="nybgxwTiTAmV" -->
## Postprocess datasets back to original shape 

Let's first generate some samples from the real data, `Xreal`, as well as some samples from our trained Generator, `Xfake`. Note that that we sample `Xreal` from the *pre-processed* dataset and not the *raw* dataset.
<!-- #endregion -->

```python colab={} colab_type="code" executionInfo={"elapsed": 2916, "status": "ok", "timestamp": 1600813861654, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="0V6Qw8WSTAmV"
n_samples= 20000
Xfake, _= generate_artificial_samples(g_model, noise_dim, n_samples)
Xreal, _= generate_real_samples(dataset, n_samples)
```

<!-- #region colab_type="text" id="oueCGXGJ2scJ" -->
Now, let's estimate the KLdivergence between `Xreal` and `Xfake`:
<!-- #endregion -->

```python
Xfake.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 434923, "status": "ok", "timestamp": 1600815185775, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="x7HfP-1P2WUD" outputId="09b66659-5049-4cdd-8198-20c31b445825"
print('Estimated KL-divergence:', KLdivergence(Xreal, Xfake, k=5, ))
```

<!-- #region colab_type="text" id="HjYK9poITAmX" -->
Let's start post-processing both `Xreal` and `Xfake`. The goal is to reverse engineer the pre-processing steps we took initially in order to convert `Xreal` back to it's *raw* state. In the meantime we apply the exact same set of transformations for `Xfake` to have the generated data in the same format as the original raw data.

We first decompose the datasets in their conntinous and discrete variables:
<!-- #endregion -->

```python colab={} colab_type="code" id="3KI-ZkMCTAmY"
# Dimensions 
dimContinuous= len(continuousColumns)
dimNominal= len(nominalColumns)

# Artificial Dataset 
Xfake_continuous= Xfake[:,0:dimContinuous] 
Xfake_nominal= Xfake[:, dimContinuous:]

# Real Dataset 
Xreal_continuous= Xreal[:, 0:dimContinuous]
Xreal_nominal= Xreal[:, dimContinuous:]
```

<!-- #region colab_type="text" id="yBkRHfT0TAma" -->
Then we start to post-process the nominal data by going from noisy marginal distributions to a clean one-hot encoded format:
<!-- #endregion -->

```python colab={} colab_type="code" id="owCHSkKwTAmb"
# Divide datasets into separated columns for each nominal variable

XrealSeparated= variableSeparator(nominalColumnValues = nominalColumnsValues, nominalDataset= Xreal_nominal)
XfakeSeparated= variableSeparator(nominalColumnValues = nominalColumnsValues, nominalDataset= Xfake_nominal)

# Function to transform from p() to ohe 
def retransformer(nominalVariable): 
  idx = nominalVariable.argmax(axis=1)
  out = np.zeros_like(nominalVariable,dtype=float)
  out[np.arange(nominalVariable.shape[0]), idx] = 1
  return out

# Transform columns into one-hot encoding format
Xreal_transformed= []
Xfake_transformed= []
for eachVariable in XrealSeparated:
    tmp= retransformer(eachVariable)
    Xreal_transformed.append(tmp)
for eachVariable in XfakeSeparated:
    tmp= retransformer(eachVariable)
    Xfake_transformed.append(tmp)


# Concatenate back into one dataset
Xreal_nominal = np.concatenate((Xreal_transformed), axis=1)
Xfake_nominal = np.concatenate((Xfake_transformed), axis=1)
```

<!-- #region colab_type="text" id="i427VxXfTAmf" -->
Next, we post-process the continous data from the range $[-1,1]$ back to their original domain:
<!-- #endregion -->

<!-- #region colab_type="text" id="Dy_Pu9nuTAmg" -->
### Real dataset 
<!-- #endregion -->

```python colab={} colab_type="code" id="j7OM3xrPTAmh"
Xreal_continuous_renormalized= ( ((Xreal_continuous + 1) / 2) * (maximum[:, None].T - minimum[:, None].T) 
                                + minimum[:, None].T)
```

<!-- #region colab_type="text" id="Y98zHOIRTAmj" -->
### Artificial dataset 
<!-- #endregion -->

```python colab={} colab_type="code" id="iH2NZNb4TAmj"
Xfake_continuous_renormalized= ( ((Xfake_continuous + 1) / 2) * (maximum[:, None].T - minimum[:, None].T) 
                                + minimum[:, None].T)
```

<!-- #region colab_type="text" id="QdcsorsCTAml" -->
Finally, we merge all post-processed data into final matrix/dataFrame:
<!-- #endregion -->

```python colab={} colab_type="code" id="Chrr9CW_TAmm"
# Real dataset
Xreal= np.concatenate( (Xreal_continuous_renormalized, Xreal_nominal), axis= 1)
# Artificial dataset 
Xfake= np.concatenate( (Xfake_continuous_renormalized, Xfake_nominal), axis= 1)
```

<!-- #region colab_type="text" id="y4VKHHbrTAmp" -->
# Evaluate the *similarity* between real and artificial distributions 

Let's recompute the KL-divergence but now for the postprocessed datasets. We see that the divergence is much larger than before. Two possible reasons: 
  
1. Our implementation of the KL-divergence estimator doesn't work properly with mixed distributions (see the KL-divergence notebook for more information) 
2. Postprocessing setup contains errors.



<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 7607, "status": "ok", "timestamp": 1599483885060, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="urlznAcyTAmp" outputId="36190b86-1f3b-4ff5-a6ca-4fba725583a7"
print('Estimated KL-divergence:', KLdivergence(Xreal, Xfake, k=10))
```

<!-- #region colab_type="text" id="nifXox5NTAmt" -->
## Visualisation of real and artificial distributions 

Let us plot the marginal continous distributions of both the real and generated data: 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 656} colab_type="code" executionInfo={"elapsed": 9787, "status": "ok", "timestamp": 1600021397451, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="KN0TfHYBTAmt" outputId="8b23a934-9e8f-49fb-aac6-e96fc884c4a9"
dim= Xreal_continuous_renormalized.shape[1]
fig= plt.figure(figsize=[15, 30])
for eachDimension in range(dim):
    plt.subplot(dim, 4, eachDimension+1)
    sns.distplot(Xreal_continuous_renormalized[:,eachDimension], bins= 100, label= 'Xreal')
    sns.distplot(Xfake_continuous_renormalized[:,eachDimension], bins= 100, label= 'Xfake')
    plt.xlabel(continuousColumns[eachDimension], fontsize = 10)
plt.legend()
plt.suptitle('Real vs. Generated Continous Marginal Distributions', fontweight= 'bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 880} colab_type="code" executionInfo={"elapsed": 6796, "status": "ok", "timestamp": 1600021399667, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="3SKpL3R4TAmx" outputId="e9c78463-3366-4cfa-a756-28c0cfd93fc6"
# Close up of some marginal distributions 

fig= plt.figure(figsize=[10,15])
plt.subplot(2,2,1)
sns.distplot(Xreal_continuous_renormalized[:,0], bins= 100, label= 'Xreal')
sns.distplot(Xfake_continuous_renormalized[:,0], bins= 100, label= 'Xfake')
plt.xlabel(continuousColumns[0], fontsize = 10)
plt.legend();

plt.subplot(2,2,2)
plt.hist(Xreal_continuous_renormalized[:, 7], bins= 100, label= 'Xreal')
plt.hist(Xfake_continuous_renormalized[:, 7], bins= 100, label= 'Xfake')
plt.xlabel(continuousColumns[7], fontsize = 10)
plt.legend();

plt.subplot(2,2,3)
plt.hist(Xreal_continuous_renormalized[:,1], bins= 100, label= 'Xreal')
plt.hist(Xfake_continuous_renormalized[:,1], bins= 100, label= 'Xfake')
plt.xlabel(continuousColumns[11], fontsize = 10)
plt.legend();

plt.subplot(2,2,4)
plt.hist(Xreal_continuous_renormalized[:,-1], bins= 100, label= 'Xreal')
plt.hist(Xfake_continuous_renormalized[:,-1], bins= 100, label= 'Xfake')
plt.xlabel(continuousColumns[-1], fontsize = 10)
plt.legend();
```

```python colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" executionInfo={"elapsed": 1889, "status": "ok", "timestamp": 1600021399670, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="FGzdbkoNTAm0" outputId="54333dac-423d-433b-d0d6-0e893fc8ae33"
# Check number of zero values 
print("Xreal -> number of zero values:", np.sum(Xreal_continuous_renormalized[:,1] == 0))
print("Xfake -> number of zero values:", np.sum(Xfake_continuous_renormalized[:,1] == 0))
```

<!-- #region colab_type="text" id="coTUKga_TAm2" -->
## Correlations between continuous variables 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 576} colab_type="code" executionInfo={"elapsed": 596, "status": "ok", "timestamp": 1600021416132, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="aY6dEzx0TAm3" outputId="6cfdcce1-e7d6-4e95-faf4-129ea420bf25"
# Correlation structure within the real dataset 
Xreal_corr= pd.DataFrame(data= np.round(np.corrcoef(Xreal_continuous_renormalized, rowvar= False), 2), 
                         columns=continuousColumns,
                        index= continuousColumns)

# Correlation structure within the fake dataset 
Xfake_corr= pd.DataFrame(data= np.round(np.corrcoef(Xfake_continuous_renormalized, rowvar= False), 2), 
                         columns=continuousColumns,
                        index= continuousColumns)

# Difference in correlation structure between the datasets 
X_corr_diff= Xreal_corr - Xfake_corr
X_corr_diff
```

<!-- #region colab_type="text" id="NnZ7wJHRTAm5" -->
## Plot nominal marginal distributions

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" executionInfo={"elapsed": 14022, "status": "ok", "timestamp": 1600021449631, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="6FXUZf21TAm5" outputId="883121a7-6e4b-4f31-b13c-e585ef4a637a"
fig= plt.figure(figsize=[10, 45])
dim= len(nominalColumnsValues)
idx_Xreal=np.arange(1,dim*2,2)
idx_Xfake=np.arange(2,dim*2+2,2)
for eachVariable in range(dim):
    

    plt.subplot(dim, 2, idx_Xreal[eachVariable])
    plt.hist(retransformer(XrealSeparated[eachVariable]), bins=[.5,.5,1.5], ec= 'k')
    plt.xlabel(nominalColumns[eachVariable])
    
    plt.subplot(dim, 2, idx_Xfake[eachVariable])
    plt.hist(retransformer(XfakeSeparated[eachVariable]), bins=[.5,.5,1.5], ec= 'k')
    plt.xlabel(nominalColumns[eachVariable])
    
    
    
plt.suptitle('Real (left) vs. Generated (right) Nominal Distributions');
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
```

<!-- #region colab_type="text" id="_2kpUTrPZPDa" -->
## We can train the model further if we believe convergence isn't reached yet:

Go back to the [evaluation section](#step5) to reevaluate the new model:
<!-- #endregion -->

```python colab={} colab_type="code" id="HNBtD22VY-yE"
wgan.fit(dataset, batch_size= batch_size, epochs= epochs, callbacks= [cbk])
```

<!-- #region colab_type="text" id="HQ9eZRrQyDyB" -->



# Save the calibrated Generator and the synthetic dataset   

If model performance is satisfactory we can go ahead and save our trained Generator together with the synthetic dataset. 
<!-- #endregion -->

<!-- #region colab_type="text" id="1yrm1DJyzPbi" -->
##  Create Pandas DataFrame of synthetic generated data:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 340} colab_type="code" executionInfo={"elapsed": 1871, "status": "error", "timestamp": 1600023220978, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="_zJrDxCLiLbu" outputId="4146ac64-fc78-4f47-cf25-514d83e12e41"
## Convert continuous synthetic data to df 

df_Xfake_continuous= pd.DataFrame(data= Xfake_continuous_renormalized, columns= continuousColumns)

## Convert nominal synthetic data to df 

#get all of the nominal variables (in one-hot encoded format)
nominalVariablesList = variableSeparator(nominalColumnValues = nominalColumnsValues, nominalDataset= Xreal_nominal)
#initialize dataframe before loop
df_Xfake_nominal= pd.DataFrame()
#Tranform from one-hot encoding to original shape for each nominal variable 
for eachColumn in range(len(nominalVariablesList )):
  #create dataFrame object 
  tmp= pd.DataFrame(nominalVariablesList[eachColumn])
  #equate column names to unique values of the variable 
  uniqueValues= datasetNominal.iloc[:,eachColumn].unique()
  uniqueValues.sort() #sort values (just like ohe of sklearn does)
  tmp.columns= uniqueValues 
  #condense one-hot encoding back to original shape 
  tmp2= tmp.idxmax(axis='columns')
  #concatenate into one dataFrame containing all of the nominal columns 
  df_Xfake_nominal= pd.concat([df_Xfake_nominal, tmp2], axis= 1)
#name the columns appropriately 
df_Xfake_nominal.columns= nominalColumns

## Merge the continuous and nominal synthetic data to one df 
df_Xfake= pd.concat([df_Xfake_continuous, df_Xfake_nominal], axis= 1)
#match the same layout of the original dataset
df_Xfake= df_Xfake[[i for i in list(FHL_bank.columns) + redudantColumns if i not in list(FHL_bank.columns) or i not in redudantColumns] ]
```

<!-- #region colab_type="text" id="GoIcULgszUzu" -->
## Save Generator and synthetic dataset either locally or on the cloud:
<!-- #endregion -->

```python
LCL= False
```

```python colab={} colab_type="code" id="M64Mwlfys6mn"
if LCL:
    ## Save the DataFrame
    df_Xfake.to_csv(r'C:\Users\ilias\Desktop\Statistics and Data Science\Jupyter Notebook\Yields.io\Stress Testing with GAN\FHL_synthetic_data.csv')
    ## Save the trained Generator 
    g_model.save(r'C:\Users\ilias\Desktop\Statistics and Data Science\Jupyter Notebook\Yields.io\Stress Testing with GAN\FHL_generator_model')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 3515, "status": "ok", "timestamp": 1599855056288, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="ecfjFpDTu2GE" outputId="79fd273a-f75d-46c3-f2b7-2c41a312256f"
if not LCL:
    ## Save the DataFrame
    df_Xfake.to_csv('/content/gdrive/My Drive/Stress Testing with GANs/data/processed/FHL_data.csv')
    ## Save the trained Generator 
    !mkdir -p saved_model
    g_model.save('/content/gdrive/My Drive/Stress Testing with GANs/mdl/FHL_generator_model')
```

# Experimental

```python colab={"base_uri": "https://localhost:8080/", "height": 231} colab_type="code" executionInfo={"elapsed": 667, "status": "error", "timestamp": 1600023863737, "user": {"displayName": "Ilias Aarab", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhzNtHScD28wke-0msegdBgEhZ-pJjf7sGQqU2c=s64", "userId": "11844581934326721507"}, "user_tz": -120} id="ynasEPqL02Uv" outputId="2dd9132e-a709-40f3-e4cd-63020f2d0707"
####### XGBOOST- IMPORT LIBRARIES NEEDED ##########

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score

#for i in range(Xfake_continuous_renormalized.shape[1]):
Xfake= Xfake_continuous_renormalized[:,:]
## truncation
#Xfake= Xfake.clip(min=0)
Xreal= Xreal_continuous_renormalized[:,:]
Xfake_new= np.ones((n_samples,Xreal_continuous_renormalized.shape[1]+1))
Xfake_new[:,:-1]= Xfake 
Xreal_new= np.zeros((n_samples,Xreal_continuous_renormalized.shape[1]+1))
Xreal_new[:,:-1]= Xreal
df_XGBOOST= np.concatenate( (Xfake_new, Xreal_new), axis=0)

## NOMINAL -> OK! 
Xfake= df_Xfake_nominal.iloc[:7500,:]
Xreal= FHL_bank[nominalColumns]
Xfake_new= np.ones((7500, Xfake.shape[1]+1))
Xfake_new[:,:-1]= Xfake.values
Xreal_new= np.zeros((7500, Xfake.shape[1]+1))
#Xreal_new[:,:-1]= Xreal.values
df_XGBOOST= np.concatenate( (Xfake_new, Xreal_new), axis=0)

X= df_XGBOOST[:,0:-1]
y= df_XGBOOST[:,-1]

seed = 1
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train, y_train);

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Variable %int: Accuracy: %.2f%%" % (i, accuracy * 100.0))
```
