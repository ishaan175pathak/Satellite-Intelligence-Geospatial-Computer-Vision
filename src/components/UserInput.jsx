import React, { useState, useRef } from 'react';

const UserInput = () => {
  const [image, setImage] = useState(null);
  const [mask, setMask] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Reference to reset the file input field
  const fileInputRef = useRef(null);

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) {
      setError('Please select an image file.');
      return;
    }

    console.log(file);
    

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setError('');

    setTimeout(() => setLoading(false), 10000)

    // creating a url file for the preview

    const ORIGINAL_IMAGE = URL.createObjectURL(file)

    try {
      const response = await fetch('http://localhost:8000/uploads', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        setImage(ORIGINAL_IMAGE)
        const data = await response.json();
        setMask(data.image);
        setError('');
      } else {
        setError('Failed to process the image.');
      }
    } catch (err) {
      setError('An error occurred while processing the image.');
    } finally {
      setLoading(false);
      // 
      // Reset the file input so the same file can be uploaded again if needed
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="relative min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto bg-white p-6 rounded shadow-md">
        <h2 className="text-xl font-semibold mb-4 text-center">Upload an Image to start</h2>
        
        <form onSubmit={(e) => e.preventDefault()}>
          <div className="mb-4">
            <label className="block text-gray-700 mb-2" htmlFor="image-upload">
              Select an image file
            </label>
            <input
              ref={fileInputRef}
              type="file"
              id="image-upload"
              accept="image/*"
              onChange={handleImageUpload}
              className="w-full p-2 border rounded file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
            />
          </div>

          {error && <p className="mt-4 text-red-500 font-medium">{error}</p>}
          
          {loading ? (
            <p className="mt-4 text-blue-500 font-medium animate-pulse">Processing image...</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {image && (
                <div className="mt-6">
                  <h3 className="text-lg font-semibold mb-2">Original Image</h3>
                  <div className="border rounded-lg overflow-hidden bg-gray-50 p-2 flex justify-center items-center">
                    <img src={image} alt="Uploaded Preview" className="max-h-80 w-auto object-contain" />
                  </div>
                </div>
              )}

              {mask && (
                <div className="mt-6">
                  <h3 className="text-lg font-semibold mb-2">Segmentation Mask</h3>
                  <div className="border rounded-lg overflow-hidden bg-gray-50 p-2 flex justify-center items-center">
                    <img src={mask} alt="Segmentation Mask" className="max-h-80 w-auto object-contain" />
                  </div>
                </div>
              )}
            </div>
          )}
        </form>
      </div>
    </div>
  )};


export default UserInput;