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

-- This is an import. `Prelude` is the name of the standard library in Haskell.
-- This is also imported automatically, but we're going to be overwriting some
-- of its entries, so we need a reference to it with a name ("P").
-- There are a bunch of different "import" syntaxes that you can find here:
-- https://wiki.haskell.org/Import
import Prelude as P

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
-- On most modern processors, this has a range of -2^63 .. 2^63-1
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

--------------------------------------------------------------------------------

-- Haskell's has two inbuilt collection types. The first is the list.
-- From a programming perspective, this is literally a singly-linked list,
-- much like you'll find in computer science textbooks.
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

-- Any type that belongs to the `Enum` Typeclass (more on those later) can
-- be used with ranges!

-- You can index into lists (though don't forget its an O(n) operation!)
oneThousand :: Integer
oneThousand = [1..] !! 999 -- 1000

-- You can also join lists:
-- This is O(n) in the size of the first (i.e. left-hand) list.
-- It will go into an infinite loop if the first list is infinite.
-- (Though it will work fine if the second one is).
oneTo6 :: [Int]
oneTo6 = [1,2,3] ++ [4,5,6]

-- We can also append single elements to the head. This is a O(1) operation
oneTo4 :: [Int]
oneTo4 = 1 : [2,3,4] -- [1,2,3,4]
-- Note that [1,2,3] : [4,5,6] wouldn't work, because the first thing
-- is also a list.

-- We also have list comprehensions (a-la Python):
doubled1to5 :: [Int]
doubled1to5 = [ x * 2 | x <- [1..5]] -- [2,4,6,8,10]

-- We can also filter in list comprehensions
withFilter :: [Int]
withFilter = [ x * x | x <- [1..10], x * x > 16] -- [25,36,49,64,81,100]

-- The other in-built collection type is the tuple.
-- Tuples group a fixed number of things together, but the things can be
-- different types.
aTuple :: (Int, Bool, String)
aTuple = (42, True, "Hello, World")

--------------------------------------------------------------------------------

-- Functions are defined similarly to other values:
-- Here, `add` is the name of the function, and `a` and `b` are the names of its
-- arguments.
add a b = a + b

-- Function type signatures look slightly different:
add :: Int -> Int -> Int
-- The `->` arrow is pronounced "to", i.e. "Int to Int to Int"
-- The types line up positionally with the arguments and return value.
-- i.e. The first type corresponds to the first argument, the second type to the
-- second argument, etc. The last type in the chain is always the return type.
-- In this sense, constants are just functions which take no arguments, and
-- always return the same value!

-- The syntax for function calls is different from most other languages.
-- Instead of using parentheses, it uses space!
callAdd :: Int
callAdd = (add 2) 3 -- 5
-- In Haskell, its normal to say functions are "applied" rather than "called".
-- So the above would be said as "add applied to 2 then 3"

-- You can also use functions in an infix position, but putting backticks (`)
-- around the function name:
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
-- There is a type and value called "unit", which is written as `()` (i.e. empty
-- parentheses). However, if a function returns that, that's the only thing it
-- can ever return.
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
-- return a string, and the other an integer).
-- In this sense, it's more similar to a ternary operator ( ? : ) in C, than a
-- normal if-then-else in most languages.

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
factorialYetAgain n          = n * factorialYetAgain (n - 1)

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
patternMatchTuples :: (Int, Int) -> (Int  , Int  )
patternMatchTuples    (x  , y  )  = (x + 1, y + 2)
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

-- There is a syntax that means you don't have to rewrite the function name
-- several times
addAllAgain :: [Int] -> Int
addAllAgain list = case list of
  []   -> 0
  x:xs -> x + addAllAgain xs
-- This is functionally identical to the first `addAll` function.

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
-- It's (sadly) nothing to do with Indian food, which is delicious.

-- In Haskell, you can pass functions as arguments to other functions.
-- Probably the most common form of this is the `map` function, which
-- takes a function and some container type (e.g. a list), and applies the
-- function to every element in the container (if any).
-- It then returns the resulting container.
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

-- Because of currying, we can pass partial functions "in line". For example:
mapExample3 :: [Int]
mapExample3 = myMap (+ 42) [1,2,3,5,7,11]
-- Will have the effect of adding each element in the supplied list to 42.
-- This is not limited to operators. Any function can be passed like this.
-- As a corollary, almost all operators in Haskell are functions (at least in
-- the sense that they have a type, and can be used in the same ways a function
-- could be used).

-- ASIDE: Haskell does not have an in-built loop construct (e.g. the `for` or
-- `while` that are often found in other languages).
-- All repetition is implemented using recursion (though this may be optimized
-- by the compiler to be loop-like under the hood).
-- There are many built-in functions which do some sort of loop. For example:
-- `fmap`, `filter`, `foldl` and `foldr`
-- It's actually very rare that you will need to do recursion by hand. For any
-- loop-like operation you need to perform, there are probably already library
-- functions you can compose together to do it for you.

--------------------------------------------------------------------------------

-- Haskell allows you to define your own types and data structures.
-- A simple example would be:
data Colour = Red | Green | Blue

-- We could then write a function which uses our new type:
say :: Colour -> String
say Red   = "Makes it go faster."
say Blue  = "Cooler than Fonzie."
say Green = "Good for the environment."

-- We can also define polymorphic data types (a.k.a. generics in other
-- languages)
data Option a = None | Some a
-- The `a` here is a type parameter. We can put any type we want in there.
-- So for example, we could have a `Option Int` or a `Option [Double]`, or even
-- a `Option Option String` (though that's a bad idea for other reasons).

-- Here's how we might use this type:
thatsLife :: Option Int
thatsLife = Some 42

nothingHere :: Option Int
nothingHere = None

-- Some terminology: `Option` is said to be the "type", while `None` and `Some`
-- are said to be the "data constructors". Somewhat confusingly, a type and
-- data constructor can have the same name. e.g. the following would be valid:
data OptionB a = Nope | OptionB a

-- The data constructors are special functions, but they are still functions.
-- You can use them anywhere you'd otherwise use a function, so long as the
-- types fit.

-- You can pattern match on these sorts of types, just like anything else.
-- Here's a `map` function for `Option`:
mapOption :: (a -> b) -> Option a -> Option b
mapOption _ None     = None
mapOption f (Some a) = Some (f a)

-- An equivalent to `Option` exists in the standard library.
-- It's called `Maybe`. It's data constructors are `Nothing` and `Just`.
-- data Maybe a = Nothing | Just a

-- For complex structures, it is preferable to use so called "record" syntax.
-- This allows you to name fields in the structure, which can then be used to
-- get and set specific fields.
data Person = Person { name :: String, height :: Double, age :: Int }
  deriving (Show)

-- The `deriving (Show)` will be explained later, but basically it allows
-- the type to be turned into a string (and thus printed to console).

-- There is also "positional" syntax. It should not be used where order matters
-- or where what goes where could be ambiguous.
data OtherPerson = OtherPerson String Double Int

-- Records can be used with positional notation, but the reverse is not true.

-- Positional construction
hanSolo :: Person
hanSolo = Person "Han Solo" 180 64

-- Record construction
obiWan :: Person
obiWan = Person { age = 57, height = 182, name = "Obi-Wan Kenobi" }

-- The field names are also functions.
-- So, for example, `name` has the type:
-- name :: Person -> String
-- You give it a Person, and it will give you their name.

-- They can also be used as setters.
-- For example
sneakyObiWan :: Person
sneakyObiWan = obiWan { name = "Ben Kenobi" }

-- Note that the original isn't modified by this. It produces a new `Person`
-- with updated record fields.

-- Haskell also allows you to alias existing types
-- For example:
type Name = String
-- This would create an alias of "String" called "Name".
-- Done this way, "String" and "Name" are compatible. You can use a "Name"
-- anywhere you can use a "String". This is chiefly useful for giving your types
-- more semantic names.

-- `newtype` by constrast creates deliberately incompatible types.
-- Here, though "Home" and "Work" represent the same sort of thing (phone
-- nubmers), you would not be able to use one where the other was expected.
newtype Home = Home String
newtype Work = Work String
data Phone = Phone Home Work

--------------------------------------------------------------------------------

-- TYPECLASSES!:

-- Haskell allows you to define so called "Typeclasses", which are essentially
-- a way of describing a set of operations that can be used on instances of that
-- Typeclass.
-- The closest analogy in other languages would be Interfaces (Java, C++, etc)
-- or Traits (Scala, Rust, etc). However, Typeclasses are much more flexible and
-- powerful than a regular Interface.

-- It's worth stating: Despite the name and syntax, Typeclasses have almost
-- nothing in common with the OO concept of a Class.

-- As an example, lets define our own `Eq` typeclass.
class Eq a where
  (==) :: a -> a -> Bool
  (!=) :: a -> a -> Bool
  a == b = not (a != b)
  a != b = not (a Examples.== b)
-- Here we've defined a typeclass called `Eq`, with two functions (==) and
-- (!=). We've given the functions some default implementations that can be
-- overwritten by instances.

-- Side Note: We need to specify "Examples.==" there because it clashes with the
-- inbuilt `==` function.
-- Similarly below, we need to specify which 'Eq' and '==' we mean.

-- Lets now declare an instance of our typeclass:
instance Examples.Eq Person where
  p1 == p2 =
    (name   p1 P.== name   p2) &&
    (height p1 P.== height p2) &&
    (age    p1 P.== age    p2)

-- Note that because of the way the default functions are defined, we did not
-- need to overwrite both of them - one was enough.
-- This is a common pattern in Haskell called the "minimal complete definition"
-- where some subset of a typeclass's functions are enough to implement the rest
-- of them.

-- Instances of typeclasses need not be "concrete" - they can themselves be
-- polymorphic (a.k.a. higher-order, or higher-kinds).
-- For instance, we can define a "Functor" typeclass. (Functor is just a fancy
-- math word for "thing you can map over").

class Functor f where
  fmap :: (a -> b) -> f a -> f b

instance Examples.Functor [] where
  fmap _ [] = []
  fmap f (x:xs) = f x : myMap f xs

instance Examples.Functor Option where
  fmap _ None = None
  fmap f (Some a) = Some (f a)

-- The really cool thing about typeclasses is that they separate type
-- declaration from the interface or operations that work on that type.
-- You can make your custom types instances of inbuilt or 3rd party typeclasses,
-- and then functions that work on instances of those typeclasses will also work
-- on your types.

-- The corollary is that you can declare your own typeclasses, and then make
-- inbuilt or 3rd party types instances, so that your functions work on them.

--------------------------------------------------------------------------------

-- Haskell lets you constrain both Functions and Typeclasses in terms of what
-- types of arguments they can take.

-- A simple example of this is sorting: To be able to put something in order,
-- you first need to be able to compare its elements with (>), (>=), (<), (<=),
-- and (==)

-- This is where the `Ord` typeclass comes in!
-- Here's its definition:

-- class P.Eq a => Ord a where
--   compare :: a -> a -> Ordering
--   (<)     :: a -> a -> Bool
--   (<=)    :: a -> a -> Bool
--   (>)     :: a -> a -> Bool
--   (>=)    :: a -> a -> Bool
--   max     :: a -> a -> a
--   min     :: a -> a -> a

-- The first thing to note here is the `P.Eq a =>`
-- This is a "type constraint". It says that a precondition of a type being an
-- instance of the typeclass "Ord" is that it is also an instance of the
-- typeclass "Eq".
-- Put another way, all Ords are Eqs, but not all Eqs are Ords.

-- We can do the same sort of thing for functions. Here's a (naive) quicksort
-- function
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (pivot:xs) =
  quicksort [x | x <- xs, x < pivot]
  ++ [pivot]
  ++ quicksort [x | x <- xs, x >= pivot]

-- The `Ord a` constraint is necessary to be able to use the comparison
-- operators.

-- We can also constrain inputs in multiple different ways.
-- For instance, this says that for all `a`, `a` must be an instance of both
-- `Num` and `Show`, while `b` need only be a member of `show`.
constraintExample :: (Num a, Show a, Show b) => a -> a -> b -> String
constraintExample x y t =
   show x ++ " plus " ++ show y ++ " is " ++ show (x+y) ++ ".  " ++ show t

-- The same goes for typeclasses!
-- Here's a diagram of some of the default Typeclasses, and their instances:
-- https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Classes.svg/480px-Classes.svg.png

-- For many of the simple type classes (e.g. Read, Show, Bounded, Enum, Eq, Ord,
-- and Functor) the Haskell compiler can automatically derive instances for you.

-- For example, we can define a binary tree like so:
data Tree a = Leaf | Node a (Tree a) (Tree a)
  deriving (Show, P.Eq, Ord)

-- And we will automatically get these functions for free
  -- show    :: (Show a) => Tree a -> String
  -- (==)    :: (Eq a)   => Tree a -> Tree a -> Bool
  -- (/=)    :: (Eq a)   => Tree a -> Tree a -> Bool
  -- compare :: (Ord a)  => Tree a -> Tree a -> Ordering
  -- (<)     :: (Ord a)  => Tree a -> Tree a -> Bool
  -- (<=)    :: (Ord a)  => Tree a -> Tree a -> Bool
  -- (>)     :: (Ord a)  => Tree a -> Tree a -> Bool
  -- (>=)    :: (Ord a)  => Tree a -> Tree a -> Bool
  -- max     :: (Ord a)  => Tree a -> Tree a -> Tree a
  -- min     :: (Ord a)  => Tree a -> Tree a -> Tree a

-- We can then compare trees - try it in the terminal.
exampleTree1 :: Tree Int
exampleTree1 = Node 7 (Node 42 Leaf Leaf) (Node 28 Leaf Leaf)

exampleTree2 :: Tree Int
exampleTree2 = Node 7 (Node 28 Leaf Leaf) (Node 42 Leaf Leaf)

-- There are also language extension that allow automatic derivation of
-- Functor, Foldable, and Traversable

--------------------------------------------------------------------------------

-- Operators and Syntactic sugar

-- Because Haskell allows you to define your own operators, its also sometimes
-- necessary to define the precedence and associativity of those operators.

-- The inbuilt operators all have a numerical precedence level and an
-- associativity (left or right) assigned to them.
-- You can see the basic ones here:
-- https://rosettacode.org/wiki/Operator_precedence#Haskell

-- You can easily find out the precedence for a specific operator by typing
-- `:i <operator you want to look up>` into GHCI.
-- For example, `:i +` will tell you that plus is "infixl 6", i.e. an infix
-- operator that is left associative, and with precedence level 6.

-- The precedence levels are between between 0 and 9, with fuctions with higher
-- precedence being applied first.
-- There is a theoretical 10th level containing function application (i.e. the
-- space operator).

-- Suppose we defined the following function:
(|||) :: (a -> Bool) -> (a -> Bool) -> a -> Bool
(|||) f g x = f x || g x

-- Then we could set its fixity like so
infixr 3 |||

-- And then use it like so:
evenOrPositive :: Integral a => a -> Bool
evenOrPositive = even ||| (> 0)

-- A couple of very useful operators are `.` and `$`
-- `$` is just function application with the lowest precedence:

($) :: (a -> b) -> a -> b
f $ x = f x

infixr 0 $

-- The most common use is to avoid using lots of brackets. So for example, an
-- expression like:
-- f (g (h x y z))
-- might instead be written as
-- f $ g $ h x y z

-- `.` is function composition (just like in maths)
-- i.e. (f . g) x == f (g x)
(.) :: (b -> c) -> (a -> b) -> a -> c
(.) f g x = f (g x)

infixr 9 .

-- Another very useful operator is <$>
-- It's the `map` operator (as in, the one we defined for `Functor`)
-- so instead of writing:
-- fmap (* 42) [1,2,3,4]
-- we could instead write
-- (* 42) <$> [1,2,3,4]

-- In this sense, `$` is function application for raw values, while `<$>` is
-- function application for functors.

-- Looked at another way, the type signature for ($) :: (a -> b) -> a -> b
-- says that `$` takes a function and gives you back the same function (albeit
-- at a lower precedence).
-- (<$>) :: (a -> b) -> f a -> f b
-- says that `<$>` takes a function and gives you back a function over whatever
-- Functor you're working with. In other words, it "lifts" the function
-- into the context of the Functor.

-- List syntax is just syntactic sugar, e.g
withSugar :: [Int]
withSugar = [1,2,3,4]

noSugar :: [Int]
noSugar = 1:2:3:4:[]

-- This also highlights another point: `:` is a function of type
-- (:) :: a -> [a] -> [a]
-- infixr 5 :

--------------------------------------------------------------------------------

-- Sub expressions

-- While Haskell is highly expression oriented - in the sense that every
-- function is really just a one-line expression - it does allow you to declare
-- named subexpressions, to make things more readable.
-- There are two syntaxes for this: `where` and `let .. in`

-- Here are some examples
quadratic :: Double -> Double -> Double -> (Double, Double)
quadratic a b c = ((negB + root) / denom, (negB - root) / denom)
  where
    negB  = -b
    root  = sqrt P.$ (b ^ (2 :: Integer)) - (4 * a * c)
    denom = 2 * a

-- We could write the above using `let` as:
quadratic2 :: Double -> Double -> Double -> (Double, Double)
quadratic2 a b c =
  let
    negB = -b
    root = sqrt P.$ (b ^ (2 :: Integer)) - (4 * a * c)
    denom = 2 * a
  in ((negB + root) / denom, (negB - root) / denom)

-- The main advantage of `let .. in` vs `where` is that `let .. in` is an
-- expression in and of itself, and can thus be the return value of a function.
