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
import CoreML

class ViewController: UIViewController {

    @IBOutlet weak var ImageButton: UIButton!
    @IBOutlet weak var TextView: UITextView!
    @IBOutlet weak var ActivityIndicator: UIActivityIndicatorView!
    
    @IBOutlet weak var AddedSugars: UILabel!
    @IBOutlet weak var NoAddedSugar: UILabel!
    
    @IBOutlet weak var SmileyIcon: UIImageView!
    @IBOutlet weak var NutritionScoreLabel: UILabel!
    @IBOutlet weak var ScoreImage: UIImageView!
    
    @IBOutlet weak var WelcomeLabel: UILabel!
    @IBOutlet weak var InstructionsLabel: UIStackView!
    
    
    let session = URLSession.shared
    
    // instantiate ML model
    let nutritionScore = NutritionScore()
    
    private let googleAPIKey = valueForAPIKey(keyname: "vision_api_key")
    var googleURL: URL {
        return URL(string: "https://vision.googleapis.com/v1/images:annotate?key=\(googleAPIKey)")!
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        //self.navigationController?.navigationBar.titleTextAttributes = [NSAttributedStringKey.font: UIFont(name: "DIN Alternate", size: 20)!]
        ActivityIndicator.isHidden = true
        AddedSugars.isHidden = true
        NoAddedSugar.isHidden = true
        TextView.isEditable = false
        SmileyIcon.isHidden = true
        NutritionScoreLabel.isHidden = true
        ScoreImage.isHidden = true
    }
    
    @IBAction func takePhoto(_ sender: Any) {
        InstructionsLabel.isHidden = true
        AddedSugars.isHidden = true
        NoAddedSugar.isHidden = true
        WelcomeLabel.isHidden = true
        TextView.text = ""
        self.SmileyIcon.isHidden = true
        NutritionScoreLabel.isHidden = true
        ScoreImage.isHidden = true
        presentImagePicker()
    }
    
    
}

// MARK: - UINavigationControllerDelegate
extension ViewController: UINavigationControllerDelegate {
}

// MARK: - UIImagePickerControllerDelegate
extension ViewController: UIImagePickerControllerDelegate {
    func presentImagePicker() {
        let imagePickerActionSheet = UIAlertController(title: "Snap Photo",
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
            ActivityIndicator.isHidden = false
            ActivityIndicator.startAnimating()
            
            // Base64 encode the image and create the request
            let binaryImageData = base64EncodeImage(selectedPhoto)
            createRequest(with: binaryImageData)
            
            dismiss(animated: true, completion: nil)
        }
    }
    
    // convert integers to doubles
    func padArray(to numToPad: Int, sequence: [NSNumber]) -> [NSNumber] {
        var newSeq = sequence
        for _ in sequence.count ... numToPad {
            newSeq.insert(NSNumber(value:0.0), at: 0)
        }
        return newSeq
    }
    
    // write ingredients as sparse vectors
    func tokenizer(words: [String]) -> [NSNumber] {
        var tokens : [NSNumber] = []
        for (index, word) in words.enumerated() {
            if top_ing.contains(word) {
                tokens.insert(NSNumber(value: 1.0), at: index)
            } else {
                tokens.insert(NSNumber(value: 0.0), at: index)
            }
        }
        return padArray(to: 999, sequence: tokens)
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
                self.ActivityIndicator.isHidden = true
                
                let labelAnnotations: JSON = responses["textAnnotations"][0]
                let numLabels: Int = labelAnnotations.count
                
                var addedSugars = ""
                if numLabels > 0 {
                    var ingredients = labelAnnotations["description"].stringValue.lowercased()
                    print("ocr output: ")
                    print(ingredients)
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
                    
                    // checking for sugars
                    for str in ingredientArray {
                        // trim any whitespace
                        let ingredient = str.trimmingCharacters(in: .whitespacesAndNewlines)
                        print(ingredient)
                        // check if ingredient contains added sugar
                        if ingDict["sugar"]!.contains(ingredient) {
                            addedSugars.append("• " + ingredient + "\n")
                        }
                    }
                    if addedSugars.isEmpty {
                        self.NoAddedSugar.isHidden = false
                        self.SmileyIcon.isHidden = false
                    }
                    else {
                        self.AddedSugars.isHidden = false
                        self.TextView.text = addedSugars
                    }
                    
                    // compute nutrition score
                    
                    // make prediction
                    let sparse_array : [NSNumber] = self.tokenizer(words: ingredientArray)
                    
                    guard let input_data = try? MLMultiArray(shape:[999], dataType:MLMultiArrayDataType.double) else {
                        fatalError("Unexpected runtime error. MLMultiArray")
                    }
                    
                    for (index,item) in sparse_array.enumerated() {
                        input_data[index] = NSNumber(floatLiteral: Double(truncating: item))                    }
                    
                    let inputs = NutritionScoreInput(sparse_ing: input_data)
                    guard let output = try? self.nutritionScore.prediction(input: inputs) else {
                        return
                    }
                    
                    let score = Int(truncating: output.score[0])
                    self.NutritionScoreLabel.isHidden = false
                    self.ScoreImage.isHidden = false
                    //self.ScoreImage.text = String(score)
                    print("score: ")
                    print(score)
                    
                    // assign score to category and select score image
                    if score <= -1 {
                        self.ScoreImage.image = UIImage(named: "A.png")
                    }
                    else if (score >= 0 && score <= 2) {
                        self.ScoreImage.image = UIImage(named: "B.png")
                    }
                    else if (score >= 3 && score <= 10) {
                        self.ScoreImage.image = UIImage(named: "C.png")
                    }
                    else if (score >= 11 && score <= 18) {
                        self.ScoreImage.image = UIImage(named: "D.png")
                    }
                    else {
                        self.ScoreImage.image = UIImage(named: "E.png")
                    }
                    
                }
                else {
                    self.TextView.text = "Oops! Please try again."
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


