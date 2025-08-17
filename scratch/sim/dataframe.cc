#include "dataframe.h"
#include "ns3/log.h" // Include log module

NS_LOG_COMPONENT_DEFINE("DataFrame"); // Define log component

// Add a column to the DataFrame
void DataFrame::addColumn(const std::string& columnName) {
    columns.push_back(columnName);
    NS_LOG_INFO("DataFrame: Added column '" << columnName << "'.");

    // Ensure all existing rows have a placeholder for the new column
    for (auto& row : data) {
        // Check if the row needs a placeholder
        if (row.size() < columns.size()) {
             row.resize(columns.size(), DataFrameElement{""}); // Add default empty string placeholder
        }
    }
}

// Add a row to the DataFrame
void DataFrame::addRow(const std::vector<DataFrameElement>& rowData) {
    if (rowData.size() == columns.size()) {
        data.push_back(rowData);
        NS_LOG_INFO("DataFrame: Added a row with " << rowData.size() << " elements.");
    } else {
        NS_LOG_ERROR("DataFrame: Error adding row. Row size (" << rowData.size() << ") does not match number of columns (" << columns.size() << "). Row not added.");
        // Optional: Handle this error more gracefully, e.g., pad the row or throw exception.
        // For now, just log error and skip the row.
    }
}

// Insert data into a specific cell
void DataFrame::insertData(long unsigned int row, long unsigned int col, const DataFrameElement& value) {
    if (row < data.size() && col < data[row].size()) {
        data[row][col] = value;
        NS_LOG_INFO("DataFrame: Inserted data into cell [" << row << "][" << col << "].");
    } else {
        NS_LOG_ERROR("DataFrame: Error inserting data. Index out of range: row=" << row << ", col=" << col << ". Max row=" << data.size() << ", Max col=" << (data.empty() ? 0 : data[0].size()));
        // Optional: Handle this error, e.g., resize or throw exception.
        // For now, just log error and do nothing.
    }
}

// Save the DataFrame to a CSV file
void DataFrame::toCsv(const std::string& filename) const {
    NS_LOG_INFO("DataFrame: Attempting to save to CSV file: '" << filename << "'.");
    std::ofstream file(filename);

    if (!file.is_open()) {
        NS_LOG_ERROR("DataFrame: Error: Could not open file '" << filename << "' for writing. Check path and permissions.");
        return;
    }

    // Write the column headers
    for (size_t i = 0; i < columns.size(); ++i) {
        file << "\"" << columns[i] << "\""; // Enclose headers in quotes
        if (i != columns.size() - 1) {
            file << ",";
        }
    }
    file << "\n";
    NS_LOG_DEBUG("DataFrame: Wrote " << columns.size() << " column headers.");

    // Write the data rows
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            // Use variantToString helper
            std::string cell_str = variantToString(row[i]);
            // Simple CSV escaping: if the string contains comma or quote, enclose in quotes and double internal quotes
            bool needs_quotes = cell_str.find(',') != std::string::npos || cell_str.find('"') != std::string::npos || cell_str.find('\n') != std::string::npos;
            if (needs_quotes) {
                std::string escaped_cell_str = cell_str;
                // Replace all " with ""
                size_t pos = escaped_cell_str.find('"');
                while (pos != std::string::npos) {
                    escaped_cell_str.replace(pos, 1, "\"\"");
                    pos = escaped_cell_str.find('"', pos + 2); // Find next after inserted ""
                }
                file << "\"" << escaped_cell_str << "\"";
            } else {
                file << cell_str;
            }


            if (i != row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    // Check if writing was successful (optional but good practice)
    if (file.fail()) {
         NS_LOG_ERROR("DataFrame: Error occurred while writing to file '" << filename << "'. File might be incomplete or corrupted.");
    } else {
         NS_LOG_INFO("DataFrame: Successfully saved " << data.size() << " rows to '" << filename << "'.");
    }
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