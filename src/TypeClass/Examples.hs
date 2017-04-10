{-# LANGUAGE DeriveFunctor #-}

module TypeClass.Examples where

import           Data.Semigroup

import           Hedgehog
import qualified Hedgehog.Gen   as Gen
import qualified Hedgehog.Range as Range

{-

True is a value:

  λ> :t True
  True :: Bool

Bool is a type:

  λ> :k Bool
  Bool :: *

We say Bool has two inhabitants:

  λ> :info Bool
  data Bool = False | True

The type () has one inhabitant:

  λ> :t ()
  () :: ()

The type Void has no inhabitants whatsoever:

  λ> :t Void

  <interactive>:1:1: error: Data constructor not in scope: Void

Just is a constructor (like True):

  λ> :t Just
  Just :: a -> Maybe a

Maybe is a type constructor taking one type argument:

  λ> :k Maybe
  Maybe :: * -> *

Either is a type constructor taking two type arguments:

  λ> :k Either
  Either :: * -> * -> *

Type application can be partially applied:

  λ> :k Either String
  Either String :: * -> *

In other words, `Maybe` and `Either String` have the same kind
This will be important later when we discuss functors.

-}

{-
Definition of a Semigroup:

  class Semigroup a where
    (<>) :: a -> a -> a

Laws:
* Associativity:         x <> (y <> z) === (x <> y) <> z

Laws are unchecked by the compiler.  It is up to the typeclass instance
to honour laws.

Laws allow for fearless refactoring.  Any expression on the LHS can be replaced
with the RHS and vice-versa without changing the meaning of the program.

Rewriting in this way may have performance implications however.

Library authors may describe rewrite rules that rewrite code according to
mechanical application of laws to improve the performance of programs.

-}

unitFormsASemigroup :: ()
unitFormsASemigroup = () <> ()

{-

Is the above valid?

Let's check the laws with property testing!

-}

unit :: Monad m => Gen.Gen m ()
unit = Gen.enumBounded

propertyUnitSemigroupAssociativity :: Property
propertyUnitSemigroupAssociativity = property $ do
  a <- forAll unit
  b <- forAll unit
  c <- forAll unit
  ((a <> b) <> c) === (a <> (b <> c))

{-

What if you have more that one associative operation for the same type?

In Haskell, newtype has no performance penatly, so introduce a new type
to disambiguate!

-}

sumFormsASemigroup :: Sum Int
sumFormsASemigroup = 2 <> 3

productFormsASemigroup :: Product Int
productFormsASemigroup = 2 <> 3

{-

Are these genuinely semigroups?

-}

propertySumOverIntSemigroupAssociativity :: Property
propertySumOverIntSemigroupAssociativity = property $ do
  a <- forAll (Sum <$> Gen.int Range.constantBounded)
  b <- forAll (Sum <$> Gen.int Range.constantBounded)
  c <- forAll (Sum <$> Gen.int Range.constantBounded)
  ((a <> b) <> c) === (a <> (b <> c))

propertyProductOverIntSemigroupAssociativity :: Property
propertyProductOverIntSemigroupAssociativity = property $ do
  a <- forAll (Product <$> Gen.int Range.constantBounded)
  b <- forAll (Product <$> Gen.int Range.constantBounded)
  c <- forAll (Product <$> Gen.int Range.constantBounded)
  ((a <> b) <> c) === (a <> (b <> c))

newtype Minus a = Minus a deriving (Eq, Show)

instance Num a => Semigroup (Minus a) where
  (Minus a) <> (Minus b) = Minus (a - b)

propertyMinusOverIntSemigroupAssociativity :: Property
propertyMinusOverIntSemigroupAssociativity = property $ do
  a <- forAll (Minus <$> Gen.int Range.constantBounded)
  b <- forAll (Minus <$> Gen.int Range.constantBounded)
  c <- forAll (Minus <$> Gen.int Range.constantBounded)
  ((a <> b) <> c) === (a <> (b <> c))

listsFormASemigroup :: [Int]
listsFormASemigroup = [1, 2] <> [3, 4]

stringsFormASemigroup :: String
stringsFormASemigroup = "Hello" <> " " <> "World"

semigroupsAreAssociative :: Bool
semigroupsAreAssociative = (2 <> (3 <> 4) :: Sum Int) == ((2 <> 3) <> 4 :: Sum Int) -- Associative law

unitHasMempty :: ()
unitHasMempty = mempty :: ()

sumHasMappend :: Sum Int
sumHasMappend = 2 `mappend` 3

productHasMappend :: Product Int
productHasMappend = 2 `mappend` 3

listHasMappend :: [Int]
listHasMappend = [1, 2] `mappend` [3, 4]

stringHasMappend :: String
stringHasMappend = "Hello" `mappend` " " `mappend` "World"

addingTwoNumbers :: Int
addingTwoNumbers = 1 + 2

addingTwoNumbersCurried :: Int
addingTwoNumbersCurried = (+) 1 2

incrementNumberWithFunctor :: Maybe Int
incrementNumberWithFunctor = (+1) <$> Just 2

incrementNumbersWithFunctor :: [Int]
incrementNumbersWithFunctor = (+1) <$> [1, 2, 3]

-- Deriving Functor.  Why does this work?
data Tree a = Leaf a | Node (Tree a) (Tree a)
  deriving (Show, Eq, Ord, Functor)

simpleTree :: Tree Int
simpleTree = Node (Node (Leaf 1) (Leaf 2)) (Leaf 3)

fmappedTree :: Tree Int
fmappedTree = (+1) <$> simpleTree

data Tree2 a = Leaf2 a | Node2 (Tree2 a) (Tree2 a)
  deriving (Show, Eq, Ord)

instance Functor Tree2 where
  f `fmap` Leaf2 a    = Leaf2 (f a)
  f `fmap` Node2 a b  = Node2 (f `fmap` a) (f `fmap` b)

simpleTree2 :: Tree2 Int
simpleTree2 = Node2 (Node2 (Leaf2 1) (Leaf2 2)) (Leaf2 3)

fmappedTree2 :: Tree2 Int
fmappedTree2 = (+1) <$> simpleTree2

functionComposition1 :: Int
functionComposition1 = let f = (+2) . (*3) in f 1

functionComposition2 :: Int
functionComposition2 = let f = (+2) . (*3) . (+4) in f 1

functionAsAFunctor1 :: Int
functionAsAFunctor1 = let f = (+2) <$> (*3) in f 1

functionAsAFunctor2 :: Int
functionAsAFunctor2 = let f = (+2) <$> (*3) <$> (+4) in f 1

prependHiWithApplicative :: Maybe String
prependHiWithApplicative = ("Hi " ++) <$> Just "Everyone"

addTwoNumbersWithApplicative1 :: Maybe Int
addTwoNumbersWithApplicative1 = (+) <$> Just 2 <*> Just 3

addTwoNumbersWithApplicative2 :: Maybe Int
addTwoNumbersWithApplicative2 = (+) <$> Nothing <*> Just 3

addTwoNumbersWithApplicative3 :: Maybe Int
addTwoNumbersWithApplicative3 = (+) <$> Just 2 <*> Nothing

addTwoNumbersWithApplicative4 :: Maybe Int
addTwoNumbersWithApplicative4 = (+) <$> Nothing <*> Nothing

addNumbersFromListWithApplicative1 :: [Int]
addNumbersFromListWithApplicative1 = (+) <$> [1] <*> [10]

addNumbersFromListWithApplicative2 :: [Int]
addNumbersFromListWithApplicative2 = (+) <$> [] <*> [10]

addNumbersFromListWithApplicative3 :: [Int]
addNumbersFromListWithApplicative3 = (+) <$> [1] <*> []

addNumbersFromListWithApplicative4 :: [Int]
addNumbersFromListWithApplicative4 = (+) <$> [] <*> []

addNumbersFromListWithApplicative5 :: [Int]
addNumbersFromListWithApplicative5 = (+) <$> [1, 2, 3] <*> [10, 20, 30]

appendTwoStringsWithApplicative :: Maybe String
appendTwoStringsWithApplicative = (++) <$> Just "Hi " <*> Just "Everyone"

functionAsAnApplicative1 :: Int
functionAsAnApplicative1 = let f = (+) <$> (+3) <*> (+4) in f 1

functionAsAnApplicative3 :: Int
functionAsAnApplicative3 = let
  g = (+) <$> (+3)
  f = g <*> (+4)
  in f 1
