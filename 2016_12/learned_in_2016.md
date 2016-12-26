# Learned in 2016

##Spider
接触爬虫不久，在跟内容提供商的接触过程中，也学到很多东西，尝试了mechanize，user_agent, selenium, 甚至被逼到用watir-webdriver做手动验证，万幸，毫无准备的情况下淌水，侥幸过了河。有些感触。  
爬虫算是功能相对封闭，单一的功能，而且在生态环境中，算是少有数据提供者，所以，做成service是比较合适的。  
直接把抓到的数据放入db中去操作，比较少组件的系统，如果要在数据下游对接很多组件，那spider应该有自己的db，然后再通过api或者其他方式来获取数据，这样方便管理，也更加清晰，同时耦合度低，降低数据库的读写压力。  

##Translate
对接一个靠谱的翻译再国内来说还是有些难度的，google对接不上，bing会是不是的被墙，在没有选择的情况下，只能做个代理了。  
AWS的服务器还是物美价廉，在国内裸着访问，速度也是不会有明显卡顿，做代理的过程中，给我的感觉是：难度明显低于我的预期，普通请求发到你的代理服务器，代理服务器再去找你想要的东西，找到之后再转发回来。  
翻译的功能相对独立，做成service也是很好的选择，不过如果翻译对db去直接操作，那么带来的负担也是很大的，需要根据业务的实际情况去调整，不过维持一张翻译表，是很有价值的做法，降低重复翻译的情况，提高运行速度，降低翻译此书，随之而来的成本和体验都会有所改观。  

## Upload
一个真正的多人实时操作的平台，难度远远高于想象。遇到了很多不同类型，不同可能的问题，可以用来开拓见识。  
首先，一个项目的架构，决定了这个项目的未来，是推倒重写还是各种补丁或者只是按计划增加功能，这些都是建立在对业务，对需求充分的熟悉之上，抽象出主次，层次，频度等等关键信息，这样去做，才会有更大的概率保证项目不会失控。这很重要，因为当你随心所欲的写到你感觉不能再写的时候，那很大概率上，就是真的不能再写了。  
项目中夹杂的比如：翻译，上传，导出等等的长请求，还有比较复杂，提交数据量较大的写操作，会让项目无论在内存还时间开销上都会有不小的负担，这是很有必要去优化的。

还有其他的东西，也是做过，接触过，不过并没有太多时间去研究，不该这样，应该是有更多时间可以用来做项目，这很不好，来年改正，我还是挺恶心的。

## 几点总结
###只加载用到的数据
最忌讳整表全数据查询，查的慢，吃内存。
###避免重复查询
从逻辑层面上进行检查，非动态数据，就用变量存起来，变量的数量也要尽可能的少否则也是很占内存的。
###适当使用索引
索引的提速是非常非常明显的，不过由于索引的原理，一张表不适合给太多字段添加索引，频繁更新的表更不适合。
###尽量少的使用数据库操作
数据库的操作速度比程序内部的操作要慢，所以减少数据库的操作次数能加快程序运行速度，当然数据库是一定要使用的，不过我们可以通过一些整理来简化，比如一个循环里每次要查一个东西，或者是要查出A再根据A来查出B，根据B查出C，这种操作都可以直接合并为一次数据库的操作，减少了db调用次数，加快的运行速度，当然是在可以结果正确的情况下。
###选择合适的结构
很多service并没有必要使用rails项目，可能使用sinatra甚至ruby，只需要很简单的几行代码就就能搞定，并不是任何时候全家桶都是最好的，全家桶对服务器的cpu和内存压力都是很大的。
###Rails本身的充分利用和最佳实践
很多时候，我们只回去使用自己觉得舒服的工具，使用的时候并不会很刻意的在意性能和内存的使用，这些代码以后必将是需要被重构的代码，当然在以项目能跑起来为第一目的的前提下，可以适当的舍弃一些代码质量，不过一定要留意和遵循一些原则，比如Rails的精选套餐里各个部分的功能的了解，使用和最佳实践，特别是那些不被人常用到的部分，这或许就是一个高手和新手的分水岭吧，让每一行代码都在她该待的地方，舒服。
###Rails的文件结构
了解这些，是知道你的代码该放在哪儿的前提。
###测试
测试很重要，举个例子来讲，你现在写的代码相当于你手里的现金，而你的测试，就相当于你银行卡里的钱，有他在，一个字，稳。