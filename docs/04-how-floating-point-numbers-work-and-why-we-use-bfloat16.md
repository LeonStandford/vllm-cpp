# 🔢 How floating-point numbers work and why we use bfloat16

Let's remind ourselves about what we want to achieve. We want to load a model. We already know the structure of the file with a model, a Safetensors file. We know our reference model architecture. We checked that model weights are stored in BF16 type. Let's take a moment to think about this type and floating-point numbers in general

Everything on the computer is binary, at the very end. [Computer numbers formats, too](https://en.wikipedia.org/wiki/Computer_number_format).

In ML models, weights are almost never integer numbers. They are rather real numbers. And computers are binary. It means that people had to figure out how to represent real numbers in programming languages, in a way that is memory efficient. In this context, it means that you can pack a lot of information in small amount of bits. Because of course, you can build a complex data type ("complex" like in "non-trivial", not like in "real+imaginary"), where the part to the left from a dot separator is stored as an integer and the part after a separator (so the less than 1 part) is stored as another integer. But you can see how inefficient is that. It's only useful for situations when you really need full precision. And this is why things like [decimal in Python](https://docs.python.org/3/library/decimal.html) exist.

Ok, so now let's think what could we do instead. The simplest alternative would be to take an integer number, like 1234 and say "I will put a dot between 12 and 34", so you'd get 12.34. People solved this with a scaling factor, like literally, for 1234 you'd use scaling factor 1/100 to get 12.34. You still need to figure out how to represent it on the computer. In this format, the fractional part (after the dot) has always fixed length - it's exactly in the middle. This format is called [fixed-point numbers](https://en.wikipedia.org/wiki/Fixed-point_arithmetic). It's less widely used than floating-point numbers

So if there are fixed-point numbers, there inevitably need to be non-fixed-point numbers. Let's just think about it, loosely. Non-fixed-point could mean that the dot (the point) can move around. The fractional part can be longer or shorter. Sounds like more memory efficient solution.

Floating-point numbers, like all other numeric types are represented as a sequences of bits. All of them have slightly different design choices, so let's focus on [regular 16-bit float](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#IEEE_754_half-precision_binary_floating-point_format:_binary16) (float16, FP16, IEEE 754-2008).

Float16 and many other floating-point numbers work in a similar way. They consists of three sections: sign, exponent and fraction. In total they take 16 bits.

```
[ sign | exponent  | fraction            ]
[ 0    | 0 1 0 0 1 | 0 0 1 0 0 0 0 0 0 0 ]
```

The intuition is like this: in floating-point numbers you can move the dot around. Fraction is the number you use to move the dot around. Like in the previous example, you fraction can be 1234. This time, dot is not fixed, so it can be anywhere you choose - 1.234, 0.0001234, 1234000 etc.

Sign is a single bit. 0 means the number is positive and 1 means negative.

Exponent is 5 bits. It controls the size of the number.

Fraction is 10 bits. It controls what number we use to move the dot around. Alternative names: significand, mantissa.

Now the formula: $(-1)^{sign} * 2^{exponent-bias} * (1.fraction)$

Notice two new things - the $1.$ before the fraction and $bias$. The `1.` is a design choice to increase the number's precision (you get 1 bit for free) without storing it explicitly in the memory. That's why it's called implicit. See this [nice Stack Overflow question and answer](https://stackoverflow.com/questions/4930269/floating-point-the-leading-1-is-implicit-in-the-significand-huh) for a better explanation. The main idea is that it's not some must-have thing, but rather a clever trick to store more data without using more memory, through saying in the floating-points numbers specification that the bit is always there (except for zero). The next new thing is $bias$ and look again where it is: $2^{exponent-bias}$. Do you have any hunches about why is it there? 

The answer is that it's there to be able to represent smaller than 1 numbers too. And specifically, to do it by making it possible to have a negative power of $2$. Exponent is always a positive number, represented as a few bits (5 bits, in float16). If we didn't subtract bias from it, $2^{exponent}$ would be always positive, and perhaps quite a big number (binary 11111 is 31 decimal). So we could generate any number bigger than 1 or smaller than -1 (almost any number, limited by the precision of floating-point number), but we couldn't represent any small but bigger than 0 number, like 0.0001234. Hence, we need subtract bias from the exponent. Bias can't be neither too big nor too small. Knowing that, what do you think the bias for float16 is? Remember that max value of exponent is 31

So the bias for float16 is 15. It means you can represent both smaller and bigger numbers.

There are more interesting things in specification of floating-point numbers, like how do we encode positive and negative infinites, how do we encode not-a-number value etc. [Wiki has many examples about exponent encoding](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Exponent_encoding)

> The sizes of exponent and fraction are main differences between different types of floating-point numbers.

Let's take an example and encode number 12.34 as float16. 12.34 is positive, so sign bit is 0. To represent 12.34, we need to turn 12 into binary:

- 12 / 2 = 6, remainder 0
- 6 / 2 = 3, remainder 0
- 3 / 2 = 1, remainder 1
- 1 / 2 = 0, remainder 1

Read it backwards and you get 1100.

So decimal 12 is binary 1100. 

Now we turn the fraction 0.34 into binary:

- 0.34 * 2 = 0.68, integer part is 0
- 0.68 * 2 = 1.36, integer part is 1, we subtract 1 and continue
- 0.36 * 2 = 0.72, integer part is 0
- 0.72 * 2 = 1.44, integer part is 1, subtract 1 and continue
- 0.44 * 2 = 0.88, integer part is 0
- 0.88 * 2 = 1.76, integer part is 1, subtract 1 and continue
- 0.76 * 2 = 1.52, integer part is 1, subtract 1 and continue
- 0.52 * 2 = 1.04, integer part is 1, subtract 1 and continue
- 0.04 * 2 = 0.08, integer part is 0
- 0.08 * 2 = 0.16, integer part is 0

At this point, we have 10 bits computed, which is our length of fraction/mantissa/significand. At the first glance, it looks like we computed exactly as many bits as we need, but it's not entirely correct. It's because we need to include the first part of the number we encode - 12 (1100 in binary) - in the mantissa. $1.$ is always an implicit part of the number, so from $1100$ we remove the first $1$. We need to use what's left - $100$ - inside the mantissa. It means that we excessively computed 3 bits, which we will discard, because of 3 bits we need to use to include $12$ in the mantissa. So we didn't make a mistake, but we did some unnecessary work. Worth remembering if you decide to ever implement your own numeric type (and if you do it by yourself, [please let me know](jedrzej@maczan.pl)!)

Btw. look at the last term: "0.08 * 2 = 0.16, integer part is 0". The process is not finished, because we didn't reach 0 (we stopped at 0.16). It means that we will lose precision when representing this number.

Reading the bits from top to bottom, decimal 0.34 is binary approximately 0101011100.

Let's put 12 and 0.34 together and we get 1100 and 0101011100. Again, $1.0$ is implicit, so we're left with 100 and 0101011100. Put them together and you have 1000101011100. That's 13 bits. Remove 3 least significant bits (the ones to the right) and the final binary representation of fraction of our number is 1000101011.

> I know I'm sidetracking a bit, you were supposed to build an inference server. But honestly, why we need to rush? If these things are not interesting to you, then you can always skip forward. I believe there's some intristic joy of learning things in depth and if you're on the same page with me, let's continue

We're done with the mantissa. Look how did we turned $1100$ into $1.100$. If we were still in decimal world (base-10), we would say that to turn $1.100$ into $1100$ we need to multiply it by $1000$ ($10^3$). The numbers we operate on are binary (base-2), so it means that we need to multiply it by $2^3$, which immediately reminds us of $2^{exponent}$ from the floating-point formula. Which means that our exponent is $3$. But, the bias. Bias is 15. And in the formula for decoding the float, in the exponent of $2$ we subtract bias from the exponent that we read from floating-point bits (these 5 bits). Which means that we need to add the bias so it shows up in the stored exponent. So we add $3+15$ and our final exponent is $18$. Let's quickly turn 18 into binary format:

- 18 / 2 = 9, remainder 0
- 9 / 2 = 4, remainder 1
- 4 / 2 = 2, remainder 0
- 2 / 2 = 1, remainder 0
- 1 / 2 = 0, remainder 1

Read backwards and decimal 18 is 10010 binary.

We can put everything together to finally see decimal 12.34 as float16:

- sign bit: 0 (positive number)
- exponent: 10010 (18 binary, 3 because we shifted left by 3 positions + 15 bias; total 5 bits)
- fraction: 1000101011 (total 10 bits)

The number of bits correctly sums to 16. Together it looks like this:

```
[ sign | exponent  | fraction            ]
[ 0    | 1 0 0 1 0 | 1 0 0 0 1 0 1 0 1 1 ]
```

One interesting thing is that we know we lost some precision when turning 12.34 into a float. It was when we computed the 0.34 part. How about verify ourselves and decode our float back to decimal to see what value computer really stores when we try to store 12.34 as a float16? You curious, too?

Recall the formula: $(-1)^{sign} * 2^{exponent-bias} * (1.fraction)$

We know the number is positive, because sign is 0, so $-1^0 = 1$

We know the exponent is 18 minus bias 15 so it's 3. Plug it into the formula and we have: $2^3=8$

Fraction is 1000101011 and we need to turn it into decimal. We need to remember also about the implicit $1.$, so all together it's $1.1000101011$. Every next digit is multiplied by a power of 2, but with an exponent - 1 than the previous one. Similar to how it works when we turn a binary into decimal - we do the same but backwards, starting from the right we multiply each digit by 2 to the power of current position. Let's check it actually: exponent 10010 is then $0 * 1 + 1 * 2 + 0 * 4 + 0 * 8 + 1 * 16 = 18$, just like we thought. Ok, so now the mantissa. We turn $1000101011$ into decimal: $1 * 2^{-1} + 0 * 2^{-2} + 0 * 2^{-3} + 0 * 2^{-4} + 1 * 2^{-5} + 0 * 2^{-6} + 1 * 2^{-7} + 0 * 2^{-8} + 1 * 2^{-9} + 1 * 2^{-10} = 0.5 + 0 + 0 + 0 + 0.03125 + 0 + 0.0078125 + 0 + 0.001953125 + 0.0009765625 = 0.5419921875$. So putting it together with implicit $1.$, our number is $1.5419921875$.

Let's plug all of them into the formula together: $1 * 8 * 1.5419921875 = 12.3359375$. Correct and close enough, at least in LLM inference. The error: $12.34 - 12.3359375 = 0.0040625$

But this was float16. And our reference model (Llama 3.2 1B Instruct) used bfloat16 (BF16). It turns out bfloat16 is widely used in many different models. It also takes 16 bits, but it's exponent is longer (8 bits) than regular float16 (5 bits). First interesting thing about bfloat16 is that the size of exponent is actually the same as of twice bigger 32-bit float (float32, FP32, float), which also has 8 bit exponent. So comparing with float16, bfloat16 trades 3 bits of fraction to gain 3 bits of exponent. So naturally, the fraction shrinks and it's only 7 bits now. So to sum up - bfloat16 is 1 sign bit, 8 bit sign exponent and 7 bit fraction.

Why bfloat16 matters? It has the same size as float16 (a half size of a full precision float), and at the same time it has the same exponent as the float32 - at the cost of smaller fraction. The industry often chooses it for inference, because it's less likely to have range issues ([overflow](https://en.wikipedia.org/wiki/Integer_overflow) / [underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow)) and at the same time the loss of precision (due to smaller fraction) is an acceptable tradeoff in LLM inference (empirical results shows it).

If you want to test your understanding, Wikipedia page about half-precision floating-point formats has [lots of good examples](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Half_precision_examples)

Now you have all prerequisities to load a model in your inference engine and start doing interesing things. Would like to read some hands-on primer on working with GPU memory in CUDA? Keep reading the next section or skip to the inference part.

Oh btw, 32-bit floats are called single-precision floats. And then you hear about double-precision. And you might be like - omg so I get TWO floating points within the same number, then? Unfortunately the double precision just means that the number is bigger (64-bit). Meh

---

[← Safetensors and your model](03-safetensors-and-your-model.md) · [🏠 Index](../README.md) · [GPU and CPU memory →](05-gpu-and-cpu-memory.md)
