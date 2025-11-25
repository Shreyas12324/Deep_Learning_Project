"""
Create a clean submission zip file for Google Classroom.
Includes only essential project files and outputs.
"""

import zipfile
import os
from pathlib import Path

def create_submission_zip():
    """Create zip file with project essentials."""
    
    # Files and folders to include
    include_patterns = [
        'app.py',
        'requirements.txt',
        'mediscan_model.h5',
        'mediscan_finetuned.h5',
        'X_train.npy',
        'X_test.npy',
        'y_train.npy',
        'y_test.npy',
        'src/*.py',
        'static/styles.css',
        'static/confusion_matrix.png',
        'static/uploads/.gitkeep',  # Keep folder structure
        'templates/*.html',
        'dataset/cataract/*.jpg',
        'dataset/cataract/*.jpeg',
        'dataset/cataract/*.png',
        'dataset/diabetic_retinopathy/*.jpg',
        'dataset/diabetic_retinopathy/*.jpeg',
        'dataset/diabetic_retinopathy/*.png',
        'dataset/glaucoma/*.jpg',
        'dataset/glaucoma/*.jpeg',
        'dataset/glaucoma/*.png',
        'dataset/normal/*.jpg',
        'dataset/normal/*.jpeg',
        'dataset/normal/*.png',
    ]
    
    # Files to explicitly exclude
    exclude_files = [
        'generate_report.py',
        'Mediscan_Project_Report.docx',
        'README.md',
        'create_submission_zip.py',
        '__pycache__',
        '.git',
        '.venv',
        'venv',
        '.gitignore',
        '.pyc'
    ]
    
    base_dir = Path.cwd()
    zip_filename = 'Mediscan_Submission.zip'
    
    print(f"Creating submission zip: {zip_filename}")
    print("Including essential project files...\n")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_STORED) as zipf:  # Use STORED for faster compression
        files_added = []
        
        # Add root level Python files
        for py_file in ['app.py', 'requirements.txt']:
            if os.path.exists(py_file):
                zipf.write(py_file, f'Mediscan/{py_file}')
                files_added.append(py_file)
                print(f"✓ Added: {py_file}")
        
        # Add model files
        for model_file in ['mediscan_model.h5', 'mediscan_finetuned.h5']:
            if os.path.exists(model_file):
                zipf.write(model_file, f'Mediscan/{model_file}')
                files_added.append(model_file)
                print(f"✓ Added: {model_file}")
        
        # Add numpy data files
        for npy_file in ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']:
            if os.path.exists(npy_file):
                zipf.write(npy_file, f'Mediscan/{npy_file}')
                files_added.append(npy_file)
                print(f"✓ Added: {npy_file}")
        
        # Add src folder (excluding report generation and pycache)
        src_dir = Path('src')
        if src_dir.exists():
            for py_file in src_dir.glob('*.py'):
                if py_file.name != '__pycache__' and not py_file.name.startswith('.'):
                    zipf.write(py_file, f'Mediscan/{py_file}')
                    files_added.append(str(py_file))
                    print(f"✓ Added: {py_file}")
        
        # Add static folder
        static_dir = Path('static')
        if static_dir.exists():
            # Add styles.css
            css_file = static_dir / 'styles.css'
            if css_file.exists():
                zipf.write(css_file, f'Mediscan/static/styles.css')
                files_added.append('static/styles.css')
                print(f"✓ Added: static/styles.css")
            
            # Add confusion matrix
            cm_file = static_dir / 'confusion_matrix.png'
            if cm_file.exists():
                zipf.write(cm_file, f'Mediscan/static/confusion_matrix.png')
                files_added.append('static/confusion_matrix.png')
                print(f"✓ Added: static/confusion_matrix.png")
            
            # Create uploads folder (with a sample or placeholder)
            uploads_dir = static_dir / 'uploads'
            if uploads_dir.exists():
                # Add a few sample uploaded images if they exist
                for img_file in list(uploads_dir.glob('*.jpg'))[:3] + list(uploads_dir.glob('*.jpeg'))[:3] + list(uploads_dir.glob('*.png'))[:3]:
                    zipf.write(img_file, f'Mediscan/{img_file}')
                    files_added.append(str(img_file))
                    print(f"✓ Added: {img_file}")
        
        # Add templates folder
        templates_dir = Path('templates')
        if templates_dir.exists():
            for html_file in templates_dir.glob('*.html'):
                zipf.write(html_file, f'Mediscan/{html_file}')
                files_added.append(str(html_file))
                print(f"✓ Added: {html_file}")
        
        # Add dataset folder (sample images from each category)
        dataset_dir = Path('dataset')
        if dataset_dir.exists():
            for category in ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']:
                category_dir = dataset_dir / category
                if category_dir.exists():
                    # Add first 10 images from each category as samples
                    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                    images_added = 0
                    for ext in image_extensions:
                        for img_file in category_dir.glob(ext):
                            if images_added >= 10:  # Limit to 10 per category to keep zip size reasonable
                                break
                            zipf.write(img_file, f'Mediscan/{img_file}')
                            images_added += 1
                        if images_added >= 10:
                            break
                    
                    if images_added > 0:
                        print(f"✓ Added: {images_added} sample images from dataset/{category}/")
    
    print(f"\n✅ Submission zip created successfully: {zip_filename}")
    print(f"📦 Total files included in zip")
    print(f"🎯 Ready for Google Classroom submission!")
    
    # Show zip file size
    zip_size = os.path.getsize(zip_filename) / (1024 * 1024)  # Convert to MB
    print(f"📊 Zip file size: {zip_size:.2f} MB")
    
    return zip_filename


if __name__ == '__main__':
    try:
        output_file = create_submission_zip()
        print(f"\n✨ Done! Upload {output_file} to Google Classroom.")
    except Exception as e:
        print(f"❌ Error creating zip: {e}")
        import traceback
        traceback.print_exc()
