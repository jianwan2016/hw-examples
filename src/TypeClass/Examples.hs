{-# LANGUAGE DeriveFunctor              #-}
{-# LANGUAGE FlexibleContexts           #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes                 #-}
{-# LANGUAGE ScopedTypeVariables        #-}

module TypeClass.Examples where

import           Control.Concurrent
import           Data.Semigroup

import           Data.Typeable
import           Hedgehog
import qualified Hedgehog.Gen       as Gen
import qualified Hedgehog.Range     as Range
import           System.IO.Unsafe

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
-- Clock as a monoid

newtype Hour = Hour Int deriving Eq

unhour :: Hour -> Int
unhour (Hour n) = ((n + 11) `mod` 12) + 1

instance Monoid Hour where
  mempty = Hour 12
  mappend (Hour a) (Hour b) = Hour ((a + b) `mod` 12)

instance Semigroup Hour

instance Show Hour where
  show h = show (unhour h) <> " O'Clock"

instance Arbitrary Hour where
  arbitrary _ = Hour <$> Gen.int (Range.linear 0 11)

{-

check (propertyMonoidAssociativityOver (Hint :: Hint Hour))
check (propertyMonoidLeftIdentityOver  (Hint :: Hint Hour))
check (propertyMonoidRightIdentityOver (Hint :: Hint Hour))

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

data Shown f = Shown String f

instance Show (Shown f) where
  show (Shown s _) = s

propertyFunctorIdentity :: (Functor f, Show (f v), Typeable f, Typeable v, Eq (f v), Arbitrary (f v)) => Hint (f v) -> Property
propertyFunctorIdentity hint = property $ do
  v <- forAll (arbitrary hint)
  fmap id v === v

propertyFunctorFusion ::
    ( Functor f, Typeable f, Typeable a
    , Show (f a), Arbitrary (f a), Eq (f c), Show (f c)
    , Typeable a, Typeable b, Typeable c
    , Show (Shown (a -> b)), Arbitrary (Shown (a -> b))
    , Show (Shown (b -> c)), Arbitrary (Shown (b -> c)))
    => Hint (f a)
    -> Hint (Shown (a -> b))
    -> Hint (Shown (b -> c))
    -> Property
propertyFunctorFusion hintFv hintAb hintBc = property $ do
  v <- forAll (arbitrary hintFv)
  (Shown _ ab) <- forAll (arbitrary hintAb)
  (Shown _ bc) <- forAll (arbitrary hintBc)
  fmap (bc . ab) v === (fmap bc . fmap ab) v

data IntToIntFunctions = IntToIntAdd | IntToIntMultiply deriving (Eq, Show, Enum, Bounded)

instance Arbitrary (Shown (Int -> Int)) where
  arbitrary _ = do
    pick :: IntToIntFunctions <- Gen.enumBounded
    case pick of
      IntToIntAdd -> do
        offset <- Gen.int (Range.linear (-10) 10)
        let description = "(+ " <> show offset <> ")"
        return $ Shown description (+ offset)
      IntToIntMultiply ->do
        offset <- Gen.int (Range.linear (-10) 10)
        let description = "(* " <> show offset <> ")"
        return $ Shown description (* offset)

{-

check (propertyFunctorIdentity  (Hint :: Hint (Maybe Int)))
check (propertyFunctorFusion  (Hint :: Hint (Maybe Int)) (Hint :: Hint (Shown (Int -> Int))) (Hint :: Hint (Shown (Int -> Int))))

-}

fmapOnMaybe1 :: Maybe Int
fmapOnMaybe1 = (+1) <$> (Just 1)

fmapOnMaybe2 :: Maybe Int
fmapOnMaybe2 = (+1) <$> Nothing

fmapOnList :: [Int]
fmapOnList = (+1) <$> [1, 2, 3]

fmapOnEither1 :: Either Int Int
fmapOnEither1 = (+1) <$> Right 1

fmapOnEither2 :: Either Int Int
fmapOnEither2 = (+1) <$> Left 1

fmapOnTuple :: (Int, Int)
fmapOnTuple = (+1) <$> (10, 20)

-- Deriving Functor.  Why does this work?
data Tree a = Leaf a | Node (Tree a) (Tree a)
  deriving (Show, Eq, Ord, Functor)

simpleTree :: Tree Int
simpleTree = Node (Node (Leaf 1) (Leaf 2)) (Leaf 3)

fmappedTree :: Tree Int
fmappedTree = (+1) <$> simpleTree

data V2 a = V2 a a deriving (Eq, Show {- , Functor -})

instance Functor V2 where
  f `fmap` (V2 a b) = V2 (f a) (f b)

data E2 a = EL a | ER a deriving (Eq, Show {- , Functor -})

instance Functor E2 where
  f `fmap` (EL a) = EL (f a)
  f `fmap` (ER a) = ER (f a)

data Tree2 a = Leaf2 a | Node2 (Tree2 a) (Tree2 a)
  deriving (Show, Eq, Ord)

instance Functor Tree2 where
  f `fmap` Leaf2 a    = Leaf2 (f a)
  f `fmap` Node2 a b  = Node2 (f `fmap` a) (f `fmap` b)

simpleTree2 :: Tree2 Int
simpleTree2 = Node2 (Node2 (Leaf2 1) (Leaf2 2)) (Leaf2 3)

fmappedTree2 :: Tree2 Int
fmappedTree2 = (+1) <$> simpleTree2

fmappedFunction :: Int
fmappedFunction = ((+1) <$> (*2)) 4

data Future a = Future (IO a) | Never

hang :: forall a . IO a
hang = do
  varA :: MVar a <- newEmptyMVar
  a <- readMVar varA
  return a

await :: Future a -> IO a
await (Future io) = io
await Never       = hang  -- block forever

newFuture :: IO (Future a, a -> IO ())
newFuture = do
  v <- newEmptyMVar
  return (Future (readMVar v), putMVar v)

future :: IO a -> Future a
future mka = unsafePerformIO $ do
  (fut, sink) <- newFuture
  _ <- forkIO $ mka >>= sink
  return fut

mkFutureDelay :: Int -> a -> Future a
mkFutureDelay delay a = future $ do
  putStrLn "Start"
  threadDelay delay
  putStrLn "Done"
  return a

{-

let !f = mkFutureDelay 10000000 (10 :: Int)
await f

-}

instance Functor Future where
  fmap f (Future get) = future (fmap f get)
  fmap _ Never        = Never

{-

let !f = mkFutureDelay 10000000 (10 :: Int)
let !g = (+1) <$> f
await g

-}

{-

But Functor only allows you to map one value in a context.  It cannot be used to
apply a function to two values in context.

For thqt we need Applicative.

-}

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

maybeApplicativeA :: Maybe Int
maybeApplicativeA = pure 2 -- Just 2

maybeApplicativeB :: Maybe Int
maybeApplicativeB = pure 3 -- Just 5

maybeApplicativeC :: Maybe (Int -> Int)
maybeApplicativeC = pure (+5) -- Just (+5)

maybeApplicativeD :: Maybe Int
maybeApplicativeD = pure (+5) <*> pure 2
               -- = pure ((+5) 2)           -- Law of Homomorphism: pure f <*> pure x === pure (f x)
               -- = pure 7                  -- Function application

maybeApplicativeE :: Maybe Int
maybeApplicativeE = (+) <$> maybeApplicativeA <*> maybeApplicativeB
              --  = (+) <$> pure 2 <*> pure 3 -- Function application
              --  = pure (+2) <*> pure 3      -- Apply fmap
              --  = pure ((+2) 3)             -- Law of Homomorphism: pure f <*> pure x === pure (f x)
              --  = pure 5                    -- Function application

add3Numbers :: Int -> Int -> Int -> Int
add3Numbers a b c = a + b + c

{-

add3Numbers 2 3 4
add3Numbers <$> Just 2 <*> Just 3 <*> Just 4

-}

maybeApplicativeF :: Maybe Int
maybeApplicativeF = (+) <$> Just 1 <*> Just 2

maybeApplicativeG :: Maybe Int
maybeApplicativeG = (+) <$> Nothing <*> Just 2

maybeApplicativeH :: Maybe Int
maybeApplicativeH = (+) <$> Just 1  <*> Nothing

maybeApplicativeI :: Maybe Int
maybeApplicativeI = (+) <$> Nothing  <*> Nothing

listApplicativeA :: [Int]
listApplicativeA = pure 1

listApplicativeB :: [Int]
listApplicativeB = (+) <$> [1] <*> [2]

listApplicativeC :: [Int]
listApplicativeC = (+) <$> [] <*> [2]

listApplicativeD :: [Int]
listApplicativeD = (+) <$> [1]  <*> []

listApplicativeE :: [Int]
listApplicativeE = (+) <$> []  <*> []

listApplicativeF :: [Int]
listApplicativeF = (+) <$> [1, 2]  <*> [10, 20]

eitherApplicativeA :: Either String Int
eitherApplicativeA = pure 1

eitherApplicativeB :: Either String Int
eitherApplicativeB = (+) <$> Right 1 <*> Right 2

eitherApplicativeC :: Either String Int
eitherApplicativeC = (+) <$> Left "Error 1" <*> Right 2

eitherApplicativeD :: Either String Int
eitherApplicativeD = (+) <$> Right 1  <*> Left "Error 2"

eitherApplicativeE :: Either String Int
eitherApplicativeE = (+) <$> Left "Error 1"  <*> Left "Error 2"


-- instance Applicative Future where
--   pure a                      = Future (pure a)
--   Future getf <*> Future getx = future (getf <*> getx)
--   _           <*> _           = Never

-- instance Monad Future where
--   return            = pure
--   Future geta >>= h = future (geta >>= force . h)
--   Never       >>= _ = Never



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
