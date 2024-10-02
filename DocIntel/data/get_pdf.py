from fpdf import FPDF, XPos, YPos

# Create instance of FPDF class
pdf = FPDF()

# Add a page
pdf.add_page()

# Set title
pdf.set_font("Helvetica", "B", 16)  # Use Helvetica instead of Arial
pdf.cell(
    200, 10, text="Sample PDF Document", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C"
)  # Use 'text' and replace 'ln=True'

# Add some text
pdf.set_font("Helvetica", size=12)  # Use Helvetica instead of Arial
pdf.multi_cell(
    0,
    10,
    text="This is a sample PDF document used to test text extraction capabilities. "
    "The goal is to convert this PDF into an image and extract the text for further processing.",
)  # Use 'text' instead of 'txt'

# Save the PDF to a file
pdf.output("sample.pdf")
