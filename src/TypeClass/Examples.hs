{-# LANGUAGE DeriveFunctor              #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes                 #-}
{-# LANGUAGE ScopedTypeVariables        #-}

module TypeClass.Examples where

import           Data.Semigroup

import           Data.Typeable
import           Hedgehog
import qualified Hedgehog.Gen   as Gen
import qualified Hedgehog.Range as Range

--------------------------------------------------------------------------------
-- Semigroup

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
unit = pure ()

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

{-
Boom!

λ> check propertyMinusOverIntSemigroupAssociativity
  ✗ <interactive> failed after 1 test and 63 shrinks.

        ┏━━ src/TypeClass/Examples.hs ━━━
    147 ┃ propertyMinusOverIntSemigroupAssociativity :: Property
    148 ┃ propertyMinusOverIntSemigroupAssociativity = property $ do
    149 ┃   a <- forAll (Minus <$> Gen.int Range.constantBounded)
        ┃   │ Minus 0
    150 ┃   b <- forAll (Minus <$> Gen.int Range.constantBounded)
        ┃   │ Minus 0
    151 ┃   c <- forAll (Minus <$> Gen.int Range.constantBounded)
        ┃   │ Minus 1
    152 ┃   ((a <> b) <> c) === (a <> (b <> c))
        ┃   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ┃   │ Failed (- lhs =/= + rhs)
        ┃   │ - Minus (-1)
        ┃   │ + Minus 1

    This failure can be reproduced by running:
    > recheck (Size 0) (Seed 5751647916938279750 2962186932628968523) <property>

False
-}

{-

What else forms a semigroup?

-}

listsFormASemigroup :: [Int]
listsFormASemigroup = [1, 2] <> [3, 4]

stringsFormASemigroup :: String
stringsFormASemigroup = "Hello" <> " " <> "World"

--------------------------------------------------------------------------------
-- Monoid

{-
Monoids are semigroups that have a special "neutral" value called `mempty` that
obey identity laws:

  class Monoid a where
    mempty  :: a
    mappend :: a -> a -> a  -- This is equivalent to (<>) in Semigroup

Laws:
* Left-Identity:             mempty `mappend` x === x
* Right-identity:            x `mappend` mempty === x
* Associativity:    x `mappend` (y `mappend` z) === (x `mappend` y) `mappend` z

-}

{-
What do you think are the neutral values for the following monoids?
-}

unitHasMempty :: ()
unitHasMempty = mempty

sumOverIntHasMempty :: Sum Int
sumOverIntHasMempty = mempty

productOverIntHasMempty :: Product Int
productOverIntHasMempty = mempty

sumHasMappend :: Sum Int
sumHasMappend = 2 `mappend` 3

productHasMappend :: Product Int
productHasMappend = 2 `mappend` 3

sumHasMConcat :: Sum Int
sumHasMConcat = mconcat [2, 3]

productHasMConcat :: Product Int
productHasMConcat = mconcat [2, 3]

listHasMappend :: [Int]
listHasMappend = [1, 2] `mappend` [3, 4]

stringHasMappend :: String
stringHasMappend = "Hello" `mappend` " " `mappend` "World"

{-

A phantom type is a type that an usued type argument.  It is typical used to carry type
information around without incurring any runtime overhead.

-}

data Hint a = Hint

class Arbitrary a where
  arbitrary :: Monad m => Hint a -> Gen.Gen m a

instance Arbitrary Int where
  arbitrary _ = Gen.int Range.constantBounded

instance Arbitrary (Sum Int) where
  arbitrary _ = Sum <$> arbitrary (Hint :: Hint Int)

instance Arbitrary (Product Int) where
  arbitrary _ = Product <$> arbitrary (Hint :: Hint Int)

propertyMonoidAssociativityOver :: (Arbitrary a, Monoid a, Eq a, Show a, Typeable a) => Hint a -> Property
propertyMonoidAssociativityOver hint = property $ do
  a :: a <- forAll (arbitrary hint)
  b :: a <- forAll (arbitrary hint)
  c :: a <- forAll (arbitrary hint)
  ((a `mappend` b) `mappend` c) === (a `mappend` (b `mappend` c))

propertyMonoidLeftIdentityOver :: (Arbitrary a, Monoid a, Eq a, Show a, Typeable a) => Hint a -> Property
propertyMonoidLeftIdentityOver hint = property $ do
  a :: a <- forAll (arbitrary hint)
  (mempty `mappend` a) === a

propertyMonoidRightIdentityOver :: (Arbitrary a, Monoid a, Eq a, Show a, Typeable a) => Hint a -> Property
propertyMonoidRightIdentityOver hint = property $ do
  a :: a <- forAll (arbitrary hint)
  (a `mappend` mempty) === a

{-

check (propertyMonoidAssociativityOver (Hint :: Hint (Sum Int)))
check (propertyMonoidAssociativityOver (Hint :: Hint (Product Int)))

check (propertyMonoidLeftIdentityOver (Hint :: Hint (Sum Int)))
check (propertyMonoidLeftIdentityOver (Hint :: Hint (Product Int)))

check (propertyMonoidRightIdentityOver (Hint :: Hint (Sum Int)))
check (propertyMonoidRightIdentityOver (Hint :: Hint (Product Int)))

-}

{-

And to show this really works, here is a case that fails.

-}

instance Arbitrary (Minus Int) where
  arbitrary _ = Minus <$> arbitrary (Hint :: Hint Int)

instance Num a => Monoid (Minus a) where
  mempty = Minus 0
  Minus a `mappend` Minus b = Minus (a - b)

{-

check (propertyMonoidAssociativityOver (Hint :: Hint (Minus Int)))
check (propertyMonoidLeftIdentityOver (Hint :: Hint (Minus Int)))
check (propertyMonoidRightIdentityOver (Hint :: Hint (Minus Int)))

-}

{-

Average as a monoid

-}

instance Arbitrary Double where
  arbitrary _ = Gen.double (Range.linearFrac (-1000) 1000)

instance Arbitrary (Sum Double) where
  arbitrary _ = Sum <$> Gen.double (Range.linearFrac (-1000) 1000)

{-

check (propertyMonoidAssociativityOver (Hint :: Hint (Sum Double)))
check (propertyMonoidLeftIdentityOver  (Hint :: Hint (Sum Double)))
check (propertyMonoidRightIdentityOver (Hint :: Hint (Sum Double)))

-}

instance Arbitrary Rational where
  arbitrary _ = toRational <$> Gen.double (Range.linearFrac (-1000) 1000)

instance Arbitrary (Sum Rational) where
  arbitrary _ = Sum . toRational <$> Gen.double (Range.linearFrac (-1000) 1000)

{-

check (propertyMonoidAssociativityOver (Hint :: Hint (Sum Rational)))
check (propertyMonoidLeftIdentityOver  (Hint :: Hint (Sum Rational)))
check (propertyMonoidRightIdentityOver (Hint :: Hint (Sum Rational)))

-}

data Average a = Average a Int deriving (Eq, Show)

instance Arbitrary a => Arbitrary (Average a) where
  arbitrary _ = Average <$> arbitrary Hint <*> Gen.int (Range.linear 0 1000)

instance Fractional a => Monoid (Average a) where
  mempty = Average 0 0
  Average a i `mappend` Average b j = Average (a + b) (i + j)

average :: Fractional a => a -> Average a
average a = Average a 1

evalAverage :: Fractional a => Average a -> Maybe a
evalAverage (Average a i) = if i > 0 then Just (a / fromIntegral i) else Nothing

averages :: Fractional a => [a] -> Average a
averages as = mconcat $ average <$> as

average1 :: Maybe Double
average1 = evalAverage $ mconcat (average <$> [1, 2, 3, 4, 5, 6])

average2 :: Maybe Double
average2 = let
    as = averages [1, 2, 3] :: Average Double
    bs = averages [4, 5]    :: Average Double
    cs = averages [6]       :: Average Double
  in evalAverage $ mconcat [as, bs, cs]

{-

check (propertyMonoidAssociativityOver (Hint :: Hint (Average Double)))
check (propertyMonoidLeftIdentityOver  (Hint :: Hint (Average Double)))
check (propertyMonoidRightIdentityOver (Hint :: Hint (Average Double)))

-}

{-

check (propertyMonoidAssociativityOver (Hint :: Hint (Average Rational)))
check (propertyMonoidLeftIdentityOver  (Hint :: Hint (Average Rational)))
check (propertyMonoidRightIdentityOver (Hint :: Hint (Average Rational)))

-}

--------------------------------------------------------------------------------
-- Data & Types

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

--------------------------------------------------------------------------------
-- Functor

addingTwoNumbers :: Int
addingTwoNumbers = 1 + 2

addingTwoNumbersCurried :: Int
addingTwoNumbersCurried = (+) 1 2



incrementNumberWithFunctor :: Maybe Int
incrementNumberWithFunctor = (+1) <$> Just 2

incrementNumbersWithFunctor :: [Int]
incrementNumbersWithFunctor = (+1) <$> [1, 2, 3]

{-
Definition of Functor:

  class Functor f where
    fmap :: (a -> b) -> f a -> f b

Laws:
* Functor-identity:     fmap id x === id x
* Fusion:            fmap (g . f) === fmap g . fmap f
-}

instance Arbitrary a => Arbitrary (Maybe a) where
  arbitrary _ = do
    value <- Gen.bool
    if value
      then Just <$> arbitrary (Hint :: Hint a)
      else pure Nothing

propertyFunctorIdentity :: (Functor f, Show (f v), Typeable f, Typeable v, Eq (f v), Arbitrary (f v)) => Hint (f v) -> Property
propertyFunctorIdentity hint = property $ do
  v <- forAll (arbitrary hint)
  fmap id v === v

{-

check (propertyFunctorIdentity  (Hint :: Hint (Maybe Int)))

-}

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

{-
Definition of Applicative:

    class Functor f => Applicative f where
      pure  :: a -> f a
      (<*>) :: f (a -> b) -> f a -> f b

Laws:
* Identity:                  pure id <*> v === v
* Homomorphism:          pure f <*> pure x === pure (f x)
* Interchange:                u <*> pure y === pure ($ y) <*> u
* Composition:  pure (.) <*> u <*> v <*> w === u <*> (v <*> w)

-}

propertyApplicativeIdentity :: (Applicative f, Show (f v), Typeable f, Typeable v, Eq (f v)) => Gen.Gen IO (f v) -> Property
propertyApplicativeIdentity genV = property $ do
  v <- forAll genV
  (pure id <*> v) === v

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
