import PyPDF2
import sys

def crop_pdf_bottom(input_path, output_path, crop_bottom=50):
    """
    Crop bottom whitespace from PDF
    
    Args:
        input_path: Path to input PDF
        output_path: Path to output PDF
        crop_bottom: Points to crop from bottom (1 point = 1/72 inch)
    """
    try:
        # Open the PDF
        with open(input_path, 'rb') as input_file:
            pdf_reader = PyPDF2.PdfReader(input_file)
            pdf_writer = PyPDF2.PdfWriter()
            
            # Process each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                
                # Get current page dimensions
                media_box = page.mediabox
                lower_left_x = float(media_box.lower_left[0])
                lower_left_y = float(media_box.lower_left[1])
                upper_right_x = float(media_box.upper_right[0])
                upper_right_y = float(media_box.upper_right[1])
                
                # Crop bottom by increasing lower_left_y
                new_lower_left_y = lower_left_y + crop_bottom
                
                # Set new crop box
                page.mediabox.lower_left = (lower_left_x, new_lower_left_y)
                page.mediabox.upper_right = (upper_right_x, upper_right_y)
                page.cropbox.lower_left = (lower_left_x, new_lower_left_y)
                page.cropbox.upper_right = (upper_right_x, upper_right_y)
                
                pdf_writer.add_page(page)
            
            # Write to output file
            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)
        
        print(f"✓ Successfully cropped PDF!")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Cropped {crop_bottom} points from bottom")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    input_pdf = r"d:\study\Papers\ACIIDS\Accids-2026\Report\figures\System3_horizontal.pdf"
    output_pdf = r"d:\study\Papers\ACIIDS\Accids-2026\Report\figures\System3_horizontal_cropped.pdf"
    
    # Crop 80 points from bottom (adjust this value as needed)
    crop_pdf_bottom(input_pdf, output_pdf, crop_bottom=120)
