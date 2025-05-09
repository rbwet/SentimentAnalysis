Wetzel and Brownell

Firstly, part 1 of this project is working perfectly. The naive-bayes algorithm is working as expected, and meeting all qualifications of the documentation. When working on part 2, some issues arose with passing the mislabelled tweets to the LLM. The labelling and prediction scores are doing what is expected of them, but it is very difficult to incorporate the right LLM outputs for some reason, even after copying the signature from lab 04 and 05. With this in mind, at our current point, about 75 percent of the project is working but the LLM prompting is not!

This could be due to several reasons: 

1. test case
    - could the test tweets genuinely not be classified as mislabelled? would this mean nothing is being passed into the analyzer? 

2. LLM usage and client 
    - is there an issue with the client key? why does the output end immediately? it is almost like there is not a handshake with the API, even though everything is flawless on our end in terms of calling the model 

3. other unforeseen issues
    - what else could be the issue here? 