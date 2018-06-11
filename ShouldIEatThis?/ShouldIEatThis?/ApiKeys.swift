//
//  ApiKeys.swift
//  imagepicker
//
//  Created by Yas on 2018-06-08.
//  Copyright © 2018 Sara Robinson. All rights reserved.
//

import Foundation

// Wrapper for obtaining keys from keys.plist
func valueForAPIKey(keyname:String) -> String {
    // Get the file path for keys.plist
    let filePath = Bundle.main.path(forResource: "keys", ofType: "plist")
    
    // Put the keys in a dictionary
    let plist = NSDictionary(contentsOfFile: filePath!)
    
    // Pull the value for the key
    let value:String = plist?.object(forKey: keyname) as! String
    
    return value
}
