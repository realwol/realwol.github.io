# What do we need in an API server

```ruby
API服务器是个很大的话题，结合个人的经验大概整理一下，Rails项目在做为API Server时，整个流程中需要的各种功能块。
```

## Route constraints
在项目中，第一个接触到请求的，应该就是路由了，所以在一开始的地方就做一些限制，降低内部请求的暴露程度，尽可能的只留出我们需要的请求。这用到很多在route中的限制条件，比如：namespace constraints only等等条件，熟练使用，提高可管理程度。

## Controller Levels
Controller都会被划分层次，一般的是：Application_controller > Base_controller > Version_controller，这是一个继承的顺序，按照这个顺序，所属controller中的代码作用域降低，这样就能很好的体现方法的作用域，同时为逻辑层的处理搭好基础。

## Request authentication
在Controller分层之后，我们可以对应在层中写入不同等级的校验方法，例如：Application中写入全局校验的方法，Base中写入API请求的校验方法，具体如每个方法中需要的参数校验等等，就可以直接放在Version中去写了。清晰明了。

## Data process
校验完毕后，就是要组装数据，这就是业务逻辑要解决的范畴，不过记住用好Rails提供的机制，在构建一个优秀API server的时候，是必不可少的。 Applicaion.rb, helper method, lib moduel, concern, service 甚至 middleware都是可以酌情使用的。

## Data formatting
数据往往需要一个既定的格式送出，所以如何在业务层高效的送出正确的数据的前提下，做到更好的代码管理和维护，就是一个比需要讨论的话题，这个针对不同的业务场景，会有不同的设置方式。

## Error handle
处理请求的错误是必不可少的一个环节，错误反馈是一个API server是否优秀的代表。一般的，反馈需要给出一个既定的错误代码以及尽可能准确定位错误的提示信息。代码层，可以封装module，来做错误处理。可以定义异常再返回或者直接返回错误代码的方式来完成，错误信息可以通过配置文本来做处理。
