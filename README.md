* ***0_SoftwareEnvironment**.  General and project-specific information about the used computing environments (programming languages, integrated development environments, package managers).*

* ***interactive_app**. Contains all items (e.g., code, documentation, results) belonging to a specific computational analysis.*

**Development of an Interactive Python Application for Machine Learning on Gene Expression Data**

This project involves creating a user-friendly multipage application using Streamlit to predict cancer status based on gene expression data. The app is designed for researchers and doctors, giving them an option to upload datasets, preprocess data, and apply machine learning models with no coding knowledge. Some features include automated feature selection, customizable machine learning workflows, and visualization tools such as Random Forest feature importances and confusion matrices. The app supports dynamic adjustments to improve prediction accuracy and aims to streamline the integration of machine learning into cancer research.

**Main contact:** 

Elias Steyaert:

elias.steyaert@hotmail.com

**Team:**

Aldo Jongejan: Supervisor

Elias Steyaert: Intern


**interactive_app:** 

This is the root directory of the app. This contains the subdirectory 'pages' and the scripts 'home.py' and 'styles.py'. Home.py is the first page of the script and will be used to start the application when you write 'streamlit run home.py' on the therminal (CLI) in the fedora38 virtual box.

**pages:** 

This folder contains the modular scripts for different parts of the computation, ensuring a separation of tasks in a logical and chronological way.

**Dependencies:** 
```
To start using the app, run the home.py script with the following command in the CLI: 'streamlit run home.py'
Once home.py is running, the user can navigate through the different pages of the app. The pages should be run in the following chronological order to ensure correct execution. The multipage app respects this order by listing the pages sequentially in the sidebar, and when the user presses the 'next page' button, the correct page will load:

home.py (Main entry point for the app)
./pages/1_data_uploading.py (Upload and transpose data)
./pages/2_volcano_plot.py (Generate a volcano plot for differential gene expression analysis, if applicable)
./pages/3_preprocessing.py (Preprocess data, including scaling, feature selection, PCA and more)
./pages/4_machine_learning.py (Train machine learning models on the preprocessed data and visualize them)
./pages/5_predicting_data.py (Make predictions based on the trained model)
```

**How to execute the code:**

You will need to boot up your virtual machine and open your terminal.

Before running the app itself, navigate on your command line to the interactive_app folder. Perform on the command line 'pip install -r requirements.txt'. After this you should be ready to start the aplication. Stay in the same directory (interactive_app) and perform 'streamlit run home.py' this should automatically boot up a browser with the required browser/interface on it.

