import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from streamlit_option_menu import option_menu
import io

# Set page title and layout
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# Initialize session state for data
if "data" not in st.session_state:
    st.session_state["data"] = None

# Sidebar menu
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: black;'>⚙️ Features</h2>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=[
            "Dashboard", "Data Collection", "Data Preprocessing",
            "Data Visualization", "Model Training",
            "Download Preprocessed Data", "Data Visualization Overview"
        ],
        icons=["house", "cloud-upload", "tools", "bar-chart-line", "robot", "download", "eye"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f8f9fa"},
            "icon": {"color": "black", "font-size": "20px"},
            "nav-link": {
                "font-size": "14px",
                "color": "black",
                "text-align": "left",
                "margin": "5px",
                "padding": "10px",
                "border-radius": "10px"
            },
            "nav-link-selected": {
                "background-color": "black",
                "color": "white",
                "font-weight": "bold"
            },
        }
    )

# 📌 **Dashboard**
import streamlit as st
import plotly.express as px

# 📌 **Dashboard Section**
if selected == "Dashboard":
    st.title("🏠 Dashboard")

    if st.session_state["data"] is not None:
        st.write("### Dataset Preview")
        st.write(st.session_state["data"].head())

        # Basic Statistics
        st.write("### Basic Statistics")
        st.write(st.session_state["data"].describe())

        # Pairplot (Scatter Matrix)
        st.write("### 📊 Interactive Pairplot Visualization")

        numeric_columns = st.session_state["data"].select_dtypes(include=['int64', 'float64']).columns.tolist()

        if len(numeric_columns) >= 2:
            fig = px.scatter_matrix(
                st.session_state["data"],
                dimensions=numeric_columns,
                title="Interactive Pairplot Visualization",
                color_discrete_sequence=["blue"]
            )
            fig.update_layout(height=700, width=1000)
            st.plotly_chart(fig)

        else:
            st.warning("⚠️ Not enough numeric columns for pairplot visualization.")

    else:
        st.warning("⚠️ No dataset available. Go to 'Data Collection' to upload data.")


# 📌 **Data Collection**
elif selected == "Data Collection":
    st.title("📂 Data Collection")

    data_source = st.radio("Choose Data Source:", ["Upload CSV File", "Use Sample Data"])

    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            st.session_state["data"] = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.write("### Dataset Preview")
            st.write(st.session_state["data"].head())

    elif data_source == "Use Sample Data":
        sample_data = {
            "Age": [25, 30, 35, 40, 45, 50, 28, 32, 38, 42, 27, 29, 34, 37, 41, 48, 52, 55, 26, 31],
            "Salary": [50000, 60000, 70000, 80000, 90000, 100000, 55000, 62000, 75000, 82000, 48000, 59000, 73000, 78000, 85000, 95000, 110000, 115000, 47000, 61000],
            "Experience": [2, 4, 6, 8, 10, 12, 3, 5, 7, 9, 1, 3, 5, 7, 9, 11, 13, 15, 2, 4],
            "Education Level": ["Bachelor", "Master", "PhD", "Bachelor", "Master", "PhD", "Bachelor", "Master", "PhD", "Bachelor",
                                "Master", "Bachelor", "PhD", "Master", "Bachelor", "PhD", "Master", "PhD", "Bachelor", "Master"],
            "Department": ["IT", "HR", "Finance", "Marketing", "IT", "HR", "Finance", "Marketing", "IT", "HR",
                           "Finance", "Marketing", "IT", "HR", "Finance", "Marketing", "IT", "HR", "Finance", "Marketing"]
        }
        st.session_state["data"] = pd.DataFrame(sample_data)
        st.success("✅ Sample data loaded successfully!")
        st.write("### Sample Dataset Preview")
        st.write(st.session_state["data"])

# 📌 **Data Preprocessing**
elif selected == "Data Preprocessing":
    if st.session_state["data"] is not None:
        st.title("🛠 Data Preprocessing")

        # Step 1: Show Dataset Info
        st.subheader("Step 1: Dataset Information")

        categorical_columns = st.session_state["data"].select_dtypes(include=["object"]).columns.tolist()

        encoding_type = st.radio("Select Encoding Type:", ["Label Encoding", "One-Hot Encoding"], key="encoding_type")

        if encoding_type == "Label Encoding":
            encoded_data = st.session_state["data"].copy()
            for col in categorical_columns:
                encoded_data[col] = encoded_data[col].astype("category").cat.codes
            st.session_state["data"] = encoded_data  # ✅ Update session state

        elif encoding_type == "One-Hot Encoding":
            if categorical_columns:  # Check if there are categorical columns
                encoded_data = pd.get_dummies(st.session_state["data"], columns=categorical_columns, drop_first=True)
                st.session_state["data"] = encoded_data  # ✅ Update session state
            else:
                st.warning("⚠ No categorical columns found for One-Hot Encoding.")

        show_info = st.checkbox("Show Dataset Info")

        if show_info:
            total_missing = st.session_state["data"].isnull().sum().sum()
            st.write(f"**Total Missing Values:** {total_missing}")

            if total_missing == 0:
                st.success("✅ No missing values found in the dataset!")

            # ✅ Show updated dataset after One-Hot Encoding
            st.write("### Updated Dataset After Encoding")
            st.write(st.session_state["data"].head())


# 📌 **Other Features (Placeholders)**
elif selected == "Data Visualization":
    st.title("📊 Data Visualization")
    st.write("Generate insightful charts and plots.")

    if st.session_state["data"] is not None:
        visualization_type = st.selectbox(
            "Select Visualization Type:",
            ["Heatmap", "Bar Graph", "Histogram", "Box Plot", "Scatter Plot","Line Chart"]
        )

        numeric_columns = st.session_state["data"].select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = st.session_state["data"].select_dtypes(include=['object']).columns.tolist()

        if visualization_type == "Heatmap":
            st.subheader("📌 Correlation Heatmap")
            plt.figure(figsize=(10, 6))
            sns.heatmap(st.session_state["data"].corr(), annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(plt)

        elif visualization_type == "Bar Graph":
            st.subheader("📌 Bar Graph")
            x_axis = st.selectbox("Select X-axis:", st.session_state["data"].columns)
            y_axis = st.selectbox("Select Y-axis:", numeric_columns)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=st.session_state["data"][x_axis], y=st.session_state["data"][y_axis])
            plt.xticks(rotation=45)
            st.pyplot(plt)

        elif visualization_type == "Histogram":
            st.subheader("📌 Histogram")
            column = st.selectbox("Select Column:", numeric_columns)
            plt.figure(figsize=(10, 6))
            sns.histplot(st.session_state["data"][column], kde=True, bins=30)
            st.pyplot(plt)

        elif visualization_type == "Box Plot":
            st.subheader("📌 Box Plot")
            column = st.selectbox("Select Column:", numeric_columns)
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=st.session_state["data"][column])
            st.pyplot(plt)
        # elif visualization_type == "Pie Chart" and categorical_columns:
        #     column = st.selectbox("Select Column:", categorical_columns)
        #     pie_data = st.session_state["data"][column].value_counts()
        #     fig = px.pie(values=pie_data, names=pie_data.index, title=f"Pie Chart of {column}")
        #     st.plotly_chart(fig)

        # elif visualization_type == "Pie Chart":
            # st.subheader("📌 Pie Chart")
            # if len(categorical_columns) > 0:
            #     column = st.selectbox("Select Column for Pie Chart:", categorical_columns)
            #     pie_data = st.session_state["data"][column].value_counts()
            #     if len(pie_data) > 1:
            #         fig = px.pie(values=pie_data, names=pie_data.index, title=f"Pie Chart of {column}",color_discrete_sequence=px.colors.qualitative.Set3,hole=0.3)
            #         fig.update_traces(textinfo='percent+label', pull=[0.1 if i == 0 else 0 for i in range(len(pie_data))])
            #
            #         st.plotly_chart(fig)
            #     else:
            #         st.warning(f"⚠️ Not enough data to create a pie chart for '{column}'.")
            # else:
            #     st.warning("⚠️ No categorical columns available for Pie Chart.")

        elif visualization_type == "Scatter Plot":
            st.subheader("📌 Scatter Plot")
            x_axis = st.selectbox("Select X-axis:", numeric_columns)
            y_axis = st.selectbox("Select Y-axis:", numeric_columns)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=st.session_state["data"][x_axis], y=st.session_state["data"][y_axis])
            st.pyplot(plt)

        if visualization_type == "Line Chart":
            st.subheader("📉 Line Chart")
            if st.session_state["data"] is not None:
                fig = px.line(st.session_state["data"], x=st.session_state["data"].index,y=st.session_state["data"].select_dtypes(include=['number']).columns)
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ No dataset available. Go to 'Data Collection' to upload data.")



elif selected == "Data Visualization Overview":
    st.title("📊 Data Visualization Overview")

    st.subheader("Overview of Visualization Tools")
    st.write("This project includes various visualization tools to help analyze datasets effectively. Below is a description of each visualization technique used:")

    st.subheader("1️⃣ Correlation Heatmap")
    st.write("Displays the correlation between numerical variables. Darker colors indicate stronger relationships. Helps identify multicollinearity and dependencies.")

    st.subheader("2️⃣ Pairplot")
    st.write("Shows relationships between multiple numerical variables. Helps visualize trends and clustering in the dataset.")

    st.subheader("3️⃣ Bar Plot")
    st.write("Used for categorical data to compare different categories. The X-axis represents categories, while the Y-axis represents counts or values.")

    st.subheader("4️⃣ Box Plot")
    st.write("Shows the distribution and outliers in numerical data. Useful for identifying skewness and variability.")

    st.subheader("5️⃣ Scatter Plot")
    st.write("Displays relationships between two numerical variables. Helps in detecting trends and patterns.")

    st.subheader("6️⃣ Pie Chart")
    st.write("Represents categorical data as proportional slices. Useful for visualizing distribution percentages.")

    st.subheader("7️⃣ Histogram")
    st.write("Represents the frequency distribution of a numerical variable. Helps understand the shape of data (normal, skewed, etc.).")

    st.subheader("8️⃣ Line Plot")
    st.write("Shows trends over time for numerical variables. Useful for time-series analysis.")

    st.subheader("🔥 Interactive Features")
    st.write("This dashboard includes interactive features such as theme toggling, column selection for custom plots, and the ability to export visualizations for further analysis.")

    st.success("📌 Use the 'Data Visualization' section to explore these charts interactively.")


elif selected == "Model Training":
    st.title("🤖 Model Training")

    if st.session_state["data"] is not None:
        # Select target column
        st.subheader("Step 1: Select Target Column")
        target_column = st.selectbox("Choose the target variable:", st.session_state["data"].columns)

        # Select features (excluding target)
        feature_columns = [col for col in st.session_state["data"].columns if col != target_column]
        X = st.session_state["data"][feature_columns]
        y = st.session_state["data"][target_column]

        # Test Size Selection
        st.subheader("Step 2: Choose Test Size")
        test_size = st.slider("Select test size (%)", min_value=0, max_value=50, value=20, step=5) / 100

        # Model Selection
        st.subheader("Step 3: Choose ML Model")
        model_option = st.selectbox(
            "Select a machine learning model:",
            ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]
        )

        # Train Model Button
        if st.button("🚀 Train Model"):
            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

             # Store Preprocessed Data in Session State
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
            st.session_state["preprocessed_data"] = pd.concat([X_train, y_train], axis=1)

            # Show Train/Test Split Sizes
            st.write(f"✅ Training Set Size: {X_train.shape[0]} samples")
            st.write(f"✅ Testing Set Size: {X_test.shape[0]} samples")
            st.success("✔️ Model trained & preprocessed data stored!")

            # Initialize Model
            model = None
            if model_option == "Logistic Regression":
                model = LogisticRegression()
            elif model_option == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_option == "Random Forest":
                model = RandomForestClassifier()
            elif model_option == "SVM":
                model = SVC()

            # Train Model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluate Model
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"🎯 Model trained successfully with an accuracy of {accuracy:.2f}!")

    else:
        st.warning("⚠️ No dataset available. Go to 'Data Collection' to upload data.")



elif selected == "Download Preprocessed Data":
    st.title("📥 Download Preprocessed Data")
    st.write("Download the cleaned dataset.")
    if "preprocessed_data" in st.session_state:
        # Preview the Preprocessed Data
        st.dataframe(st.session_state["preprocessed_data"].head())

        # Convert DataFrame to CSV
        csv_data = st.session_state["preprocessed_data"].to_csv(index=False).encode('utf-8')

        # Add Download Button
        st.download_button(
            label="📥 Download Preprocessed Data",
            data=csv_data,
            file_name="preprocessed_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No preprocessed data available. Train a model first.")
