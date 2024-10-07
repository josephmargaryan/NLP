import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Create instance of FPDF class
pdf = FPDF()

# Add a page
pdf.add_page()

# Set title
pdf.set_font("Helvetica", "B", 16)
pdf.cell(
    200, 10, text="Sample PDF Document", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C"
)

# Add some text
pdf.set_font("Helvetica", size=12)
pdf.multi_cell(
    0,
    10,
    text="This is a sample PDF document used to test table extraction capabilities.",
)

# Add table header
pdf.set_font("Helvetica", "B", 12)
pdf.cell(40, 10, "Column 1", border=1)
pdf.cell(40, 10, "Column 2", border=1)
pdf.cell(40, 10, "Column 3", border=1)
pdf.cell(40, 10, "Column 4", border=1)
pdf.ln()

# Add table rows
pdf.set_font("Helvetica", size=12)
for i in range(1, 6):
    pdf.cell(40, 10, f"Row {i} Col 1", border=1)
    pdf.cell(40, 10, f"Row {i} Col 2", border=1)
    pdf.cell(40, 10, f"Row {i} Col 3", border=1)
    pdf.cell(40, 10, f"Row {i} Col 4", border=1)
    pdf.ln()

# Get current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Save the PDF to the current directory
output_path = os.path.join(current_directory, "sample_with_table.pdf")
pdf.output(output_path)

print(f"PDF saved to {output_path}")