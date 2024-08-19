import streamlit as st
import pandas as pd
from PIL import Image
import torch
from tqdm import trange
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import pandas as pd
import torch.nn as nn
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
sns.set_style("whitegrid")


from cadlae.detector import AnomalyDetector, DetectorHelper
model = AnomalyDetector()
def intro():
    
    #TODO increase the introduction page to give overview of entire framework
    st.write("# Welcome to CADLAE 👋")

    st.markdown(
        """
        Configurable Anomaly Detection, Localization, and Explanation Framework for Cyber Physical Systems
        
        **👈 Select a demo from the dropdown on the left to explore the framework**
        
        ## About CADLAE
        This thesis paper addresses the issue of anomaly detection in cyber-physical systems (CPS), which are systems that combine physical and computational components to monitor and control physical processes. In CPS, anomaly detection is a crucial task, as it can help prevent system failures, ensure safety, and optimise performance. Anomalies in CPS can arise due to a range of reasons, including sensor malfunctions, faults in communication channels, and cybersecurity attacks.

        ### CADLAE Framework
        """)
    image = Image.open("images/architecture.png")
    # make image 0.75 times the size of the original
    image = image.resize((int(image.width * 0.75), int(image.height * 0.75)))
    st.image(image, use_column_width=True)
    st.markdown("""
        ### Detection
        #### Seq-2-Seq LSTM Encoder Decoder
        """)
    st.markdown(
        """
        To address the problem of anomaly detection, the proposed method in this thesis is an Seq2Seq LSTM Autoencoder
        model. An autoencoder is an unsupervised learning method that learns to encode data into a lower-dimensional
        space and then reconstruct it. In this approach, the encoder encodes the input data, and the decoder
        reconstructs it back to its original form. The LSTM (Long Short-Term Memory) component of the model allows
        for the detection of temporal dependencies, which are essential for detecting anomalies in time series data,
        as in CPS.
        """)
    image = Image.open("images/lstm.png")
    # make image 0.75 times the size of the original
    image = image.resize((int(image.width * 0.75), int(image.height * 0.75)))
    st.image(image, use_column_width=True)
    st.markdown("""### Localization""")
    st.markdown("""
        The proposed method uses the feature-wise error generated from model predictions to localise anomalies. The feature-wise error measures the difference between the actual value of a feature and its predicted value by the model. The proposed method applies several techniques to this error to localise the anomaly. These include PCA (Principal Component Analysis) localisation and thresholding localisation.
        """)
    st.markdown("""#### PCA Localisation""")
  
    st.markdown("""
        PCA localisation is a technique that attempts to reduce the dimensionality of the feature-wise error space by projecting it onto a lower-dimensional subspace. This technique is applied to subgraphs, which are groups of highly correlated features that are used to detect and localise anomalies. The purpose of this technique is to identify the features that are contributing the most to the anomaly.
        """)
    st.markdown("""#### Thresholding Localisation""")
    st.markdown("""
        Thresholding localisation is another technique used in the proposed method, which involves setting a threshold for the feature-wise error. The threshold is set based on the maximum feature-wise error in the training set. Any feature whose error exceeds this threshold is considered anomalous and is localised.
        """)
    st.markdown("""#### Subgraph Detection""")
    st.markdown("""
        Localising anomalies to a group of features rather than a single one is essential in CPS as features are often highly interdependent. Identifying a single anomalous feature may not be enough to determine the root cause of the anomaly, as it may be influenced by other features in the system. The proposed method attempts to identify a group of highly correlated features that are grouped together using their correlation to form disjoint subgraphs.
        """)
    st.markdown("""### Explanation""")
    st.markdown("""
        The proposed approach uses an unsupervised model to generate predictions, which are then used to train a
		supervised explanation model. The explanation model leverages Gradient Boosting Machines (GBMs) to
		generate interpretable and actionable rules in the form of if-then statements.
		GBMs are a popular machine learning model that iteratively adds decision trees to the model to
		improve prediction accuracy.
		""")
    
    st.markdown("""
		The GBM model can be fitted to the training data by minimizing the loss function using gradient descent.
		GBMs can be interpreted by examining the importance of each feature in the model, which is calculated by
		measuring how much the model's accuracy decreases when a feature is randomly shuffled. GBMs also
		provide information about the contribution of each feature to each decision tree in the model.
		""")
   
    st.markdown(
        '''
        ### Evaluation
        #### The Tennessee Eastman Process
        '''
    )
    image = Image.open("images/TEP_diagram.png")
    # make image 0.75 times the size of the original
    image = image.resize((int(image.width * 0.75), int(image.height * 0.75)))
    # load image with boarder around it

    st.image(image, use_column_width=True)
    st.markdown(
        '''
        The Tennessee Eastman process has been widely used as a testbed to study various challenges faced in continuous
        processes. Originally proposed by Downs and Vogel (1993), the TEP has been used for plant-wide control design,
        multivariate control, optimisation, predictive control, adaptive control, nonlinear control, process diagnostics
        , and educational purposes.In recent years, many studies involving the TEP have focused on fault detection using
        classical statistics or machine learning methods.
        '''
    )
    st.markdown(
        """
		### Relevant Links

		- Find the accompanying [Thesis](https://github.com/CameronLooney)
		- View the demo [Source Code](https://github.com/CameronLooney)
		- View the CADLAE [Source Code](https://github.com/CameronLooney)
		- Find me on [LinkedIn](https://www.linkedin.com/in/cameronlooney/)
	"""
    )

def train_and_predict_page():
    from cadlae.detector import AnomalyDetector
    from page.train_page import prediction
    prediction()
    

        

      

        
        
        
            
        
        
    
    
        
        
    


def plotting_demo():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!
"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")

def correlation_subgraph():
    from page.corr_subgraph_page import generate_corr_subgraph
    generate_corr_subgraph()
def data_frame_demo():
    import streamlit as st
    import pandas as pd
    import altair as alt

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo shows how to use `st.write` to visualize Pandas DataFrames.

(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
"""
    )

    @st.cache_data
    def get_UN_data():
        AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
        df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
        return df.set_index("Region")

    try:
        df = get_UN_data()
        countries = st.multiselect(
            "Choose countries", list(df.index), ["China", "United States of America"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write("### Gross Agricultural Production ($B)", data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

def localise_pca():
    from page.local_pca_page import generate_pca_localisation
    generate_pca_localisation()
    
def localise_threshold():
    from page.local_threshold import generate_threshold_localisation
    generate_threshold_localisation()
    
def explainer():
    from page.explainer_page import generation_explanation
    generation_explanation()
    
def feedback_form():
    from page.feedback_form import feedback
    feedback()
    
def full_demo():
    from page.full_pipeline import pipeline
    pipeline()
page_names_to_funcs = {
    "Introduction": intro,
    "Full Demo": full_demo,
    "Train Model": train_and_predict_page,
    "Correlation Subgraph": correlation_subgraph,
    "PCA Localisation": localise_pca,
    "Threshold Localisation": localise_threshold,
    "Explanation" : explainer,
    "Feedback Form" : feedback_form,
    
   
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()