//
//  ViewController.swift
//  ShouldIEatThis?
//
//  Created by Mary Hoekstra on 2018-06-08.
//  Copyright © 2018 Mary Hoekstra. All rights reserved.
//

import Foundation
import UIKit
import SwiftyJSON

class ViewController: UIViewController {

    @IBOutlet weak var ImageButton: UIButton!
    @IBOutlet weak var TextView: UITextView!
    @IBOutlet weak var ActivityIndicator: UIActivityIndicatorView!
    
    let session = URLSession.shared
    
    private let googleAPIKey = valueForAPIKey(keyname: "vision_api_key")
    var googleURL: URL {
        return URL(string: "https://vision.googleapis.com/v1/images:annotate?key=\(googleAPIKey)")!
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        ActivityIndicator.isHidden = true
    }
    
    @IBAction func takePhoto(_ sender: Any) {
        presentImagePicker()
    }
    
}

// MARK: - UINavigationControllerDelegate
extension ViewController: UINavigationControllerDelegate {
}

// MARK: - UIImagePickerControllerDelegate
extension ViewController: UIImagePickerControllerDelegate {
    func presentImagePicker() {
        let imagePickerActionSheet = UIAlertController(title: "Snap/Upload Image",
                                                       message: nil, preferredStyle: .actionSheet)
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            let cameraButton = UIAlertAction(title: "Take Photo",
                                             style: .default) { (alert) -> Void in
                                                let imagePicker = UIImagePickerController()
                                                imagePicker.delegate = self
                                                imagePicker.sourceType = .camera
                                                self.present(imagePicker, animated: true)
            }
            imagePickerActionSheet.addAction(cameraButton)
        }
        let libraryButton = UIAlertAction(title: "Choose Existing",
                                          style: .default) { (alert) -> Void in
                                            let imagePicker = UIImagePickerController()
                                            imagePicker.delegate = self
                                            imagePicker.sourceType = .photoLibrary
                                            self.present(imagePicker, animated: true)
        }
        imagePickerActionSheet.addAction(libraryButton)
        let cancelButton = UIAlertAction(title: "Cancel", style: .cancel)
        imagePickerActionSheet.addAction(cancelButton)
        present(imagePickerActionSheet, animated: true)
    }
    func imagePickerController(_ picker: UIImagePickerController,
                               didFinishPickingMediaWithInfo info: [String : Any]) {
        if let selectedPhoto = info[UIImagePickerControllerOriginalImage] as? UIImage {
            ActivityIndicator.startAnimating()
            
            // Base64 encode the image and create the request
            let binaryImageData = base64EncodeImage(selectedPhoto)
            createRequest(with: binaryImageData)
            
            dismiss(animated: true, completion: nil)
        }
    }
}

extension ViewController {
    
    func analyzeResults(_ dataToParse: Data) {
        
        // Update UI on the main thread
        DispatchQueue.main.async(execute: {
            
            // Use SwiftyJSON to parse results
            let json = JSON(dataToParse)
            let errorObj: JSON = json["error"]
            
            // Check for errors
            if (errorObj.dictionaryValue != [:]) {
                self.TextView.text = "Error code \(errorObj["code"]): \(errorObj["message"])"
            } else {
                // Parse the response
                let responses: JSON = json["responses"][0]
                self.ActivityIndicator.stopAnimating()
                
                let labelAnnotations: JSON = responses["textAnnotations"][0]
                let numLabels: Int = labelAnnotations.count
                
                var addedSugars = ""
                if numLabels > 0 {
                    var ingredients = labelAnnotations["description"].stringValue.lowercased()
                    //print("ocr output: ")
                    //print(ingredients)
                    print("---------------------------------------")
                    
                    // extract string starting at "ingredients" and ending with a period
                    print("testing")
                    let range = ingredients.range(of: "ingredients")
                    let startIndex = range?.lowerBound
                    let newString = String(ingredients[startIndex!...])
                    let periodIndex = newString.range(of: ".\n")
                    let endIndex = periodIndex?.lowerBound
                    let endIndexBeforePeriod = newString.index(endIndex!,offsetBy: -1)
                    ingredients = String(newString[...endIndexBeforePeriod])
                    print(ingredients)
                    print("\n---------------------------------------\n")
                    
                    // clean up ingredients string
                    ingredients = ingredients.replacingOccurrences(of: "ingredients", with: "")
                    ingredients = ingredients.replacingOccurrences(of: "\n", with: " ")
                    // replace brackets with a comma so sub-ingredients are still parsed
                    ingredients = ingredients.replacingOccurrences(of: " (", with: ", ")
                    ingredients = ingredients.replacingOccurrences(of: ")", with: "")
                    ingredients = ingredients.replacingOccurrences(of: " [", with: ", ")
                    ingredients = ingredients.replacingOccurrences(of: "]", with: "")
                    // replace other list seperators with a comma
                    ingredients = ingredients.replacingOccurrences(of: ": ", with: ", ")
                    ingredients = ingredients.replacingOccurrences(of: "; ", with: ", ")
                    ingredients = ingredients.replacingOccurrences(of: ". ", with: ", ")
                    // strip words like "including" and "contains" for better results
                    ingredients = ingredients.replacingOccurrences(of: "including ", with: "")
                    ingredients = ingredients.replacingOccurrences(of: "contains ", with: "")
                    ingredients = ingredients.replacingOccurrences(of: "*", with: "")
                    
                    // split ingredients string on commas
                    var ingredientArray = ingredients.components(separatedBy: ", ")
                    ingredientArray = Array(Set(ingredientArray))
                    for str in ingredientArray {
                        // trim any whitespace
                        let ingredient = str.trimmingCharacters(in: .whitespacesAndNewlines)
                        print(ingredient)
                        // check if ingredient contains added sugar
                        if ingDict["sugar"]!.contains(ingredient) {
                            addedSugars.append(ingredient + " (added sugar)\n")
                        }
                    }
                    self.TextView.text = addedSugars
                }
                    
                else {
                    self.TextView.text = "No ingredients found"
                }
                
            }
        })
        
    }
    
    func resizeImage(_ imageSize: CGSize, image: UIImage) -> Data {
        UIGraphicsBeginImageContext(imageSize)
        image.draw(in: CGRect(x: 0, y: 0, width: imageSize.width, height: imageSize.height))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        let resizedImage = UIImagePNGRepresentation(newImage!)
        UIGraphicsEndImageContext()
        return resizedImage!
    }
}


/// Networking
extension ViewController {
    func base64EncodeImage(_ image: UIImage) -> String {
        var imagedata = UIImagePNGRepresentation(image)
        
        // Resize the image if it exceeds the 2MB API limit
        if ((imagedata?.count)! > 2097152) {
            let oldSize: CGSize = image.size
            let newSize: CGSize = CGSize(width: 800, height: oldSize.height / oldSize.width * 800)
            imagedata = resizeImage(newSize, image: image)
        }
        
        return imagedata!.base64EncodedString(options: .endLineWithCarriageReturn)
    }
    
    func createRequest(with imageBase64: String) {
        // Create our request URL
        var request = URLRequest(url: googleURL)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue(Bundle.main.bundleIdentifier ?? "", forHTTPHeaderField: "X-Ios-Bundle-Identifier")
        
        // Build our API request
        let jsonRequest = [
            "requests": [
                "image": [
                    "content": imageBase64
                ],
                "features": [
                    [
                        "type": "DOCUMENT_TEXT_DETECTION",
                        "maxResults": 1
                    ]
                ],
                "imageContext": [
                    "languageHints": "en"
                ]
            ]
        ]
        let jsonObject = JSON(jsonRequest)
        
        // Serialize the JSON
        guard let data = try? jsonObject.rawData() else {
            return
        }
        
        request.httpBody = data
        
        // Run the request on a background thread
        DispatchQueue.global().async { self.runRequestOnBackgroundThread(request) }
    }
    
    func runRequestOnBackgroundThread(_ request: URLRequest) {
        // run the request
        
        let task: URLSessionDataTask = session.dataTask(with: request) { (data, response, error) in
            guard let data = data, error == nil else {
                print(error?.localizedDescription ?? "")
                return
            }
            
            self.analyzeResults(data)
        }
        
        task.resume()
    }
}


