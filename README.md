### Processing



<u>*Explanation*</u>:

*This is the default GitHub README.md file. Do not change its name.*



*Subdirectories*

* ***0_SoftwareEnvironment**.  General and project-specific information about the used computing environments (programming languages, integrated development environments, package managers).*

* ***Data**. Raw, meta, and pre-processed data*

* ***NameOfComputation_1**.  Placeholder (change to appropriate name. E.g., Preprocessing, Simulation, etc). Contains all items (e.g., code, documentation, results) belonging to a specific computational analysis.*

  

*<u>Instructions</u>:* 

* *Remove all Instructions and the Explanation once you have completed the template.*

* *Level of detail: Information provided should be sufficient for someone who was not involved in the project and/or has limited knowledge about the topic,  to understand and reproduce the project.* 

  

* *Change 'NameOfComputation_1' to a more descriptive name. Make additional \NameOfComputation_1 sub-directories if required.*
  
  * *Different parts of the computational work should be placed in different \NameOfComputation_1 directories to keep the \Processing modular.* 
  * *You may add any prefix (e.g., number, date) to order these directories.*



* *The template below may duplicate (parts of) the 0_PROJECT.md file but, more importantly, should provide a short but clear description of all \NameOfComputation_1 directories. In particular, it should make the dependencies (if any) clear, and should provide instructions of how to execute the code.* 
  * *Since the FSS is the entry point of a project and because the FSS will be shared with peers (not the GitHub repository), it is not necessary (but allowed) to provide a project description in this README file*



*GitHub*

*The Processing directory should be synchronized to the corresponding GitHub repository **without data and results** because there may not be sufficient storage space on GitHub to accommodate this.  Moreover, data and results already reside in the FSS. The FSS remains the point of entry*

* *To exclude specific subdirectories and files from the GitHub repository you should make the appropriate changes to **gitignore-FSS-template** and copy it to **.gitignore**.* 
* *The \Processing directory contains several other gitignore templates for specific languages. To use any of these templates, simply merge its content into .gitignore.  It is considered good practice to keep a single .gitignore in the top-level directory and not in individual subdirectories, which would make debugging more troublesome.*

* *The \Processing directory should contain a <u>github.txt</u> file with the name of  the associated GitHub repository (URL). This file is used by the FSS Navigator*



====== TEMPLATE STARTS HERE ======

**Short title + project description:** [optional]
Development of an Interactive Python Application for Machine Learning on Gene Expression Data
This project involves creating a user-friendly multipage application using Streamlit to predict cancer status based on gene expression data. The app is designed for researchers and doctors, giving them an option to upload datasets, preprocess data, and apply machine learning models with no coding knowledge. Some features include automated feature selection, customizable machine learning workflows, and visualization tools such as Random Forest feature importances and confusion matrices. The app supports dynamic adjustments to improve prediction accuracy and aims to streamline the integration of machine learning into cancer research.

**Main contact:** [optional]
Elias Steyaert:
elias.steyaert@hotmail.com

**Team:** [optional]
Aldo Jongejan: Supervisor
Elias Steyaert: Intern


**interactive_app**:  This is the root directory of the app. This contains the subdirectory 'pages' and the scripts 'home.py' and 'styles.py'. Home.py is the first page of the script and will be used to start the application when you write 'streamlit run home.py' on the therminal (CLI) in the fedora38 virtual box.

**pages**:  This folder contains the modular scripts for different parts of the computation, ensuring a separation of tasks in a logical and chronological way.


**Dependencies:** 
To start using the app, run the home.py script with the following command in the CLI:
streamlit run home.py
Once home.py is running, the user can navigate through the different pages of the app. The pages should be run in the following chronological order to ensure correct execution. The multipage app respects this order by listing the pages sequentially in the sidebar, and when the user presses the 'next page' button, the correct page will load:

home.py (Main entry point for the app)
./pages/1_data_uploading.py (Upload and transpose data)
./pages/2_volcano_plot.py (Generate a volcano plot for differential gene expression analysis, if applicable)
./pages/3_preprocessing.py (Preprocess data, including scaling, feature selection, PCA and more)
./pages/4_machine_learning.py (Train machine learning models on the preprocessed data and visualize them)
./pages/5_predicting_data.py (Make predictions based on the trained model)

**How to execute the code:**
You will need to boot up your virtual machine and open your terminal.
Before running the app itself, navigate on your command line to the interactive_app folder. Perform on the command line 'pip install -r requirements.txt'. After this you should be ready to start the aplication. Stay in the same directory (interactive_app) and perform 'streamlit run home.py' this should automatically boot up a browser with the required browser/interface on it.

