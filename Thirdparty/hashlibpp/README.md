This is a copy of [hashlib++](http://hashlib2plus.sourceforge.net/), version 0.3.4 (13 Oct 2011).

hashlib++ is a simple and very easy to use library to create a cryptographic checksum called "hash" in C++. hashlib++ is written in plain C++ and should work with every compiler and platform. The library is released under the BSD-license and therefore free software.

hashlib++ allows you to create a hash (for example MD5 or SHA1) with only one single method-call. So if you want to create a hash based on a file or a string quickly you might be interested. Just give it a try.


```
#include <hashlibpp.h>
#include <string>
#include <iostream>  //for "cerr"

/*
 * creating a wrapper object
 */
hashwrapper *myWrapper = new md5wrapper();

try {
        /*
         * create a hash from a string
         */
        std::string hash1 = myWrapper->getHashFromString("Hello World");

        /*
         * create a hash based on a file
         */
        std::string hash2 = myWrapper->getHashFromFile("README.TXT");
}
catch(hlException &e) {
        std::cerr << "Error(" 
                  << e.error_number()
                  << "): "
                  << e.erro_message()
                  << std::endl;
}

delete myWrapper;
```