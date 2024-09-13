# Streamlit App with Snowflake Integration

This repository contains a Streamlit application integrated with Snowflake. The app connects to a Snowflake database, performs SQL queries, and visualizes the data directly in the browser.

## Features

- **Snowflake Integration**: Fetch data from Snowflake and display it within the Streamlit app.
- **Data Visualization**: Use Streamlit's built-in charting features to create interactive graphs.
- **User Input**: Allow users to input SQL queries or parameters to fetch data from Snowflake.
- **Real-Time Updates**: Display data and visualizations in real-time as users interact with the app.

## Prerequisites

Before you can run the app, ensure you have the following:

1. **Snowflake Account**: A Snowflake account with necessary privileges to read data.
2. **Snowflake Connector for Python**: Installed via pip.
3. **Streamlit**: Installed via pip.
4. **Python 3.7+**.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/streamlit-snowflake-app.git
cd streamlit-snowflake-app
```

### 2. Set up Snowflake Connection Credentials

Ensure you have the following details for connecting to Snowflake:

- **Snowflake Username**
- **Snowflake Password**
- **Account URL**
- **Warehouse**
- **Database**
- **Schema**


## Usage

1. Navigate to the app in your browser.
2. Enter the required SQL query or select predefined options to fetch data from Snowflake.
3. View the results and visualizations based on the Snowflake data.

## Configuration

You can customize the SQL queries or the interface by editing the `app.py` file. For more advanced customization, update the Snowflake connection parameters or enhance the Streamlit app layout.
This app is connected to "VCCH"."CHB_HRES" Database.
You need to ask for role "CLD-PLC-HRES-CHB-APPROLE-DATAREADER-SG" for accessing the data.


## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributing

Feel free to contribute to this project by creating issues or submitting pull requests. Please follow the [contribution guidelines](CONTRIBUTING.md).

---

This README assumes your app connects to Snowflake and lets users input queries or select pre-built options. Let me know if you'd like to add or customize anything!