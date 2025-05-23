#include "dataframe.h"
#include "ns3/log.h" // Include log module

NS_LOG_COMPONENT_DEFINE("DataFrame"); // Define log component

// Add a column to the DataFrame
void DataFrame::addColumn(const std::string& columnName) {
    columns.push_back(columnName);
    NS_LOG_DEBUG("DataFrame: Added column '" << columnName << "'.");

    // Ensure all rows have the same number of columns
    for (auto& row : data) {
        row.push_back(""); // Add default empty string as a placeholder
    }
}

// Add a row to the DataFrame
void DataFrame::addRow(const std::vector<DataFrameElement>& rowData) {
    if (rowData.size() == columns.size()) {
        data.push_back(rowData);
        NS_LOG_DEBUG("DataFrame: Added a row with " << rowData.size() << " elements.");
    } else {
        NS_LOG_ERROR("DataFrame: Error adding row. Row size (" << rowData.size() << ") does not match number of columns (" << columns.size() << ").");
    }
}

// Insert data into a specific cell
void DataFrame::insertData(long unsigned int row, long unsigned int col, const DataFrameElement& value) {
    if (row < data.size() && col < data[row].size()) {
        data[row][col] = value;
        // NS_LOG_DEBUG("DataFrame: Inserted data into cell [" << row << "][" << col << "]."); // Too verbose for large dataframes
    } else {
        NS_LOG_ERROR("DataFrame: Error inserting data. Index out of range: row=" << row << ", col=" << col << ".");
    }
}

// Save the DataFrame to a CSV file
void DataFrame::toCsv(const std::string& filename) const {
    NS_LOG_INFO("DataFrame: Attempting to save to CSV file: '" << filename << "'.");
    std::ofstream file(filename);

    if (!file.is_open()) {
        NS_LOG_ERROR("DataFrame: Error: Could not open file '" << filename << "' for writing.");
        return;
    }

    // Write the column headers
    for (size_t i = 0; i < columns.size(); ++i) {
        file << columns[i];
        if (i != columns.size() - 1) {
            file << ",";
        }
    }
    file << "\n";
    NS_LOG_DEBUG("DataFrame: Wrote " << columns.size() << " column headers.");

    // Write the data rows
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << variantToString(row[i]);
            if (i != row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    NS_LOG_INFO("DataFrame: Successfully saved " << data.size() << " rows to '" << filename << "'.");
}

// Get the number of columns in the DataFrame
int DataFrame::getNumColumns() const {
    return columns.size();
}

// Get the number of rows in the DataFrame
int DataFrame::getNumRows() const {
    return data.size();
}

// Helper function to convert variant to string for output
std::string DataFrame::variantToString(const DataFrameElement& element) const {
    std::ostringstream oss;
    std::visit([&oss](auto&& value) {
        oss << value;
    }, element);
    return oss.str();
}