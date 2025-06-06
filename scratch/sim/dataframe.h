#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <variant> // To store mixed types
#include <sstream>
#include <cstdint>
#include <algorithm> // For std::min, std::max, std::visit (in C++17)

// Define a variant that can hold several types (int, double, string, uint32_t, etc.)
// Added long unsigned int as it appears in insertData parameters.
using DataFrameElement = std::variant<int, double, std::string, uint32_t, long unsigned int, bool, float, int64_t, uint64_t>; // Expanded types if needed


class DataFrame {
public:
    // Constructor
    DataFrame() = default;

    // Function to add a column
    void addColumn(const std::string& columnName);

    // Function to add a row
    void addRow(const std::vector<DataFrameElement>& rowData);

    // Function to insert data into a specific cell (row, column)
    void insertData(long unsigned int row, long unsigned int col, const DataFrameElement& value);

    // Function to save the DataFrame to a CSV file
    void toCsv(const std::string& filename) const;

    // Function to get number of columns
    int getNumColumns() const;

    // Function to get number of rows
    int getNumRows() const;

private:
    std::vector<std::string> columns; // Column names
    std::vector<std::vector<DataFrameElement>> data; // Data rows with mixed types

    // Helper function to convert variant to string
    std::string variantToString(const DataFrameElement& element) const;
};
