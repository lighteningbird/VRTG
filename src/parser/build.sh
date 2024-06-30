git clone https://github.com/tree-sitter/tree-sitter-go
cd tree-sitter-go && git checkout 3a3a8ee53777eaa6cbfc032afd11af407ee1f7fe && cd ..
git clone https://github.com/tree-sitter/tree-sitter-javascript
cd tree-sitter-javascript && git checkout 5720b249490b3c17245ba772f6be4a43edb4e3b7 && cd ..
git clone https://github.com/tree-sitter/tree-sitter-python
cd tree-sitter-python && git checkout dd3861c8f433c963cab727ad0827b55da74faae6 && cd ..
git clone https://github.com/tree-sitter/tree-sitter-ruby
cd tree-sitter-ruby && git checkout f257f3f57833d584050336921773738a3fd8ca22 && cd ..
git clone https://github.com/tree-sitter/tree-sitter-php
cd tree-sitter-php && git checkout d38adb26304d9b9d38e9a3b4aae0ec4b29bf9462 && cd ..
git clone https://github.com/tree-sitter/tree-sitter-java
cd tree-sitter-java && git checkout c194ee5e6ede5f26cf4799feead4a8f165dcf14d && cd ..
git clone https://github.com/tree-sitter/tree-sitter-c-sharp
cd tree-sitter-c-sharp && git checkout 1648e21b4f087963abf0101ee5221bb413107b07 && cd ..
git clone https://github.com/tree-sitter/tree-sitter-c
cd tree-sitter-c && git checkout 84bdf409067676dd5c003b2a7cb7760456e731bf && cd ..
python build.py
