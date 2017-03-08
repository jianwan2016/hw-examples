-- This is a comment.
{- This
   is
   a
   multiline
   comment
 -}

-- This is a module definition.
-- By default this will export everything in the module
-- (i.e. make it available to other modules)
module Examples where

-- If we wanted to limit what we export, we could use:
-- module Examples (things, we, want, to, export) where

-- This is a constant named "three", with value `3`
-- Identifiers for values and functions in Haskell follow the usual
-- rules, but must start with lower-case letter, and must not start with a
-- leading underscore. They can also contain some unusual characters, such as
-- apostrophes.
three = 3

-- Haskell will automatically infer types in most cases. However, we can provide
-- a specific type ourselves.
-- The `::` syntax roughly means "has the type"
three :: Int

-- Identifiers are immutable. e.g. I cannot now set `three` to another number or
-- type. (It would produce a compile-time error).
-- three = "Hello" -- Compiler says no!

-- Haskell has the usual assortment of primitive types:
true :: Bool
true = True
false :: Bool
false = False

-- Fixed size signed integer.
-- On most modern processes, this is in range -2^63 .. 2^63-1
-- But it might be narrower. Be careful!
-- If you need specific sized integers, they exist (e.g. Int8, Int16, Int32)
int :: Int
int = 64

-- The usual IEEE double. (Float is also available)
double :: Double
double = 3.141592653 -- There's also a built-in constant called `pi`.

-- Character
q :: Char
q = 'q'

-- Strings
string :: String -- `String` is a synonym for [Char] (read "list of characters")
string = "Hello, World"

-- Haskell also has unbounded integers (i.e. they can represent any number, so
-- long as there is sufficient memory to store it.)
-- (The inline-types are just to supress some warnings. They're not strictly
-- necessary).
aBigNumber :: Integer
aBigNumber = (2 :: Integer) ^ (1024 :: Integer)

-- Sadly, there is no in-built equivalent for floating point numbers,
-- although there are ones that will give you rationals (Rational), and
-- arbitrarily large fixed-point reals (FixedPoint).

-- Haskell's has two inbuilt collection types. The first is the list.
-- From a programming perspective, this is literally a singly-linked list,
-- much like you'll in computer science textbooks.
-- That means no O(1) indexing!
-- Lists store 0 or more values of the same type. (No mixing types in lists!)
aList :: [Int]
aList = [1,2,3,4,5]

-- Haskell lets you do some cool things with lists. For example, ranges:
oneTo100 :: [Int]
oneTo100 = [1..100]

oneHundredTo1 :: [Int]
oneHundredTo1 = [100, 99..1] -- [100, 99, 98, 97,  ..., 1]

evens :: [Int]
evens = [0,2..100] -- [0, 2, 4, ..., 100]

aToZ :: String
aToZ = ['A'..'Z'] -- "ABCD..XYZ"

-- Infinite lists! This will give you a list of all the natural numbers
infiniteList :: [Integer]
infiniteList = [1..] -- [1,2,3,...]

-- You can index into lists (though don't forget its an O(n) operation!)
oneThousand :: Integer
oneThousand = [1..] !! 999 -- 1000

-- You can also join lists:
-- This is O(n) in the size of the first (i.e. left-hand) list.
-- It will go into an infinite loop if the first list is infinite.
-- (Though it will work fine if the second one is).
oneTo6 :: [Int]
oneTo6 = [1,2,3] ++ [4,5,6]

-- We can also just append to the head. This is a O(1) operation
oneTo4 :: [Int]
oneTo4 = 1 : [2,3,4] -- [1,2,3,4]

-- We also have list comprehensions (a-la Python):
doubled1to5 :: [Int]
doubled1to5 = [ x * 2 | x <- [1..5]] -- [2,4,6,8,10]

-- We can also filter in list comprehensions
withFilter :: [Int]
withFilter = [ x * x | x <- [1..10], x*x > 16] -- [25,36,49,64,81,100]

-- The other in-built collection type is the tuple.
-- Tuples group a fixed number of things together, but the things can be
-- different types.
aTuple :: (Int, Bool, String)
aTuple = (42, True, "Hello, World")

-- Functions are defined similarly to other values:
-- Here, `add` is the name of the function, and `a` and `b` are the names of its
-- arguments.
add a b = a + b

-- Function type signatures look slightly different:
add :: Int -> Int -> Int
-- The `->` arrow is pronounced "to", i.e. "Int to Int to Int"
-- The types line up positionally with the arguments and return value.
-- i.e. The first type corresponds to the first argument, the second type to the
-- second argument, etc. The last type in the chain is always the return value.
-- In this sense, constants are just functions which take no arguments, and
-- always return the same value!

-- The syntax for function calls is different from most other languages.
-- Instead of using parentheses, it uses space!
callAdd :: Int
callAdd = add 2 3 -- 6
-- In Haskell, its normal to say functions are "applied" rather than "called".
-- So the above would be said as "add applied to 2 and 3"

-- You can also use functions in an infix position, but putting backticks (`)
-- around the function name
infixAdd :: Int
infixAdd = 7 `add` 11

-- Haskell also lets you define your own operators. These are just regular
-- functions, but they go in the infix position by default.
-- They aren't allowed to contain letters or numbers, only symbols, and must
-- be surrounded by parentheses when you define their type
-- Here's one for modulo:
(%) :: Int -> Int -> Int
a % b = mod a b

-- If you want, you can use an operator in the prefix position (like a normal
-- function) by surrounding it with parentheses:
prefixMod :: Int
prefixMod = (%) 7 3 -- 1

-- ASIDE: Functions in Haskell must return something. They also can't return
-- `null` or `None` or `nil` like in most other languages (there's no such
-- thing in Haskell!). The closest thing to `null` conceptually is the `Maybe`
-- type, which we'll get to later.
-- Functions also have to do the same thing every time you give them the same
-- arguments. `2 + 2` can't return `4` the first time you do it, and delete your
-- database the second time. It must always return `4`.

-- Haskell has several ways to define conditionals and branching functions
-- The first is a simple if-then-else construct
factorial :: Int -> Int
factorial n =
  if n <= 1
    then 1
    else n * factorial (n - 1)
-- The main thing to note here is that the `else` is mandatory, and both the
-- `then` and `else` block must have the same type. (e.g. You can't have one
-- return a string, and the other an integer)

-- Another way is to to use guards. These are similar to an if-then-else chain:
factorialAgain :: Int -> Int
factorialAgain n
  | n <= 1    = 1
  | otherwise = n * factorialAgain (n - 1)
-- `otherwise` isn't special syntax. It's just another name for `True`.
-- You can have as many guards as you want (including only one!)
-- The compiler will warn you if it can work out that you haven't covered all
-- cases.

-- Some people advocate writing the above like this:
factorialYetAgain :: Int -> Int
factorialYetAgain n | n <= 1 = 1
factorialYetAgain n          = n * factorialAgain (n - 1)

-- The third way is pattern matching:
naiveFibonacci :: Int -> Int
naiveFibonacci 1 = 1
naiveFibonacci 2 = 1
naiveFibonacci n = naiveFibonacci (n - 1) + naiveFibonacci (n - 2)
-- This will check each "case" in turn. i.e. First it will check if the input is
-- 1, and return 1 if it is. If it's not, it will check 2. If that fails
-- it will fall through to the "general" case.

-- Pattern matching can be done one almost any type, and is very powerful!
-- Here's pattern matching on tuples:
patterMatchTuples :: (Int, Int) -> (Int  , Int  )
patterMatchTuples    (x  , y  )  = (x + 1, y + 2)
-- This is very useful for deconstructing complex types. Here we use it to break
-- a tuple up into its parts, naming the first part `x` and the second part `y`.

-- We can do similar things for lists:
addAll :: [Int] -> Int
addAll [] = 0
addAll (x:xs) = x + addAll xs
-- Here we first check if the list is empty `[]`, and if so, return 0.
-- If it's not empty, we fall through to the general case. There, we name the
-- head of the list `x`, and the tail (i.e. remainder) of the list `xs`.
-- ASIDE: If you're using a linter, it will probably tell you that `addAll`
-- can be rewritten using the `foldr` function.

-- You can pattern match on your own types, too! (We'll get to that later.)

-- Haskell has great support for polymorphism (i.e. functions that can work on
-- many different types).
-- Here's a simple polymorphic identity function, which just throws back
-- whatever you give it.
myId :: a -> a
myId foo = foo
-- The `a`s in the type signature basically mean "Put anything here".
-- However, all the `a`s have to have the same type. You can't give it an `Int`
-- and ask it to return a `Sprocket`.

-- You can write functions where the polymorphic parameters don't have to have
-- the same type. For instance, here's a function which takes two arguments,
-- throws one away, and then gives you back the other one.
myConst :: a -> b -> a
myConst foo _ = foo
-- Here, the types that go in `a` and `b` don't need to be the same (though they
-- could be!)
-- We also used an underscore `_` to say "We don't care about this argument".

-- Both `id` and `const` (equivalent to the above) are defined in the standard
-- library, and are surprisngly useful!

-- Haskell functions are "curried" by default. This means that they take their
-- first argument, and give you back a function which takes their second
-- argument, which gives you back a function which takes their third arugument,
-- and so on. Once a function has gotten all its arguments, it gives back a
-- value.
-- The most useful result of this is that we don't have to give a function all
-- its arguments at once! For example:
add42 :: Int -> Int
add42 = add 42
-- This gives us a function which will add 42 to whatever you give it.

-- ASIDE: The term "curry" comes from a guy called Haskell Curry, who invented
-- the concept. The language Haskell is also named after him.
-- It's (sadly) nothing to do with Indian food =(

-- In Haskell, you can pass functions as arguments to other functions.
-- Probably the most common form of this is the `map` function, which
-- takes a function and some container type (e.g. a list), and applies the
-- function to every element in the list (if any).
-- It then returns the resulting list.
myMap :: (a -> b) -> [a] -> [b]
myMap _    []     = []
myMap func (x:xs) = func x : myMap func xs
-- The (a -> b) in the type of `myMap` is the type of the function argument.

-- We could use this with our above `add42` example:
mapExample :: [Int]
mapExample = myMap add42 [1,2,3] -- [43, 44, 45]

-- Haskell also has lambdas, which are useful when you don't want to define a
-- full function. However, they otherwise work just the same as functions, and
-- have the same types!
-- Here's the above example with a lambda instead:
mapExample2 :: [Int]
mapExample2 = myMap (\x -> x + 42) [1,2,3]
-- If you need a lambda that takes multiple arguments, you can write it like so:
-- (\x y -> x + y)
