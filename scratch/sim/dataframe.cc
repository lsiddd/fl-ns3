#include "dataframe.h"

// Add a column to the DataFrame
void DataFrame::addColumn(const std::string& columnName) {
    columns.push_back(columnName);

    // Ensure all rows have the same number of columns
    for (auto& row : data) {
        row.push_back(""); // Add default empty string as a placeholder
    }
}

// Add a row to the DataFrame
void DataFrame::addRow(const std::vector<DataFrameElement>& rowData) {
    if (rowData.size() == columns.size()) {
        data.push_back(rowData);
    }

    else {
        std::cerr << "Error: Row size does not match number of columns" << std::endl;
    }
}

// Insert data into a specific cell
void DataFrame::insertData(long unsigned int row, long unsigned int col, const DataFrameElement& value) {
    if (row < data.size() && col < data[row].size()) {
        data[row][col] = value;
    }

    else {
        std::cerr << "Error: Index out of range" << std::endl;
    }
}

// Save the DataFrame to a CSV file
void DataFrame::toCsv(const std::string& filename) const {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file" << std::endl;
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
