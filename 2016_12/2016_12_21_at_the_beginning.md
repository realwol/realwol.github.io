# What do we need in an API server

```
API服务器是个很大的话题，结合个人的经验大概整理一下，Rails项目在做为API Server时，整个流程中需要的各种功能块。
```

## Route constraints
在项目中，第一个接触到请求的，应该就是路由了，所以在一开始的地方就做一些限制，降低内部请求的暴露程度，尽可能的只留出我们需要的请求。这用到很多在route中的限制条件，比如：namespace constraints only等等条件，熟练使用，提高可管理程度。
例如：
  
```
 # API

  namespace :api do
    namespace :v2 do
      resources :rubygems, param: :name, only: [], constraints: { name: Patterns::ROUTE_PATTERN } do
        resources :versions, param: :number, only: :show, constraints: {
          number: /#{Gem::Version::VERSION_PATTERN}(?=\.json\z)|#{Gem::Version::VERSION_PATTERN}/
        }
      end
    end

    namespace :v1 do
      resource :api_key, only: :show do
        put :reset
      end
      resources :profiles, only: :show
      resources :downloads, only: :index do
        get :top, on: :collection
        get :all, on: :collection
      end
      constraints id: Patterns::ROUTE_PATTERN, format: /json|yaml/ do
        get 'owners/:handle/gems',
          to: 'owners#gems',
          as: 'owners_gems',
          constraints: { handle: Patterns::ROUTE_PATTERN },
          format: true
```
这是rubygems.org中的一段路由，用到了很多路由控制的方法，虽然不是尽然尽美，但绝对好过我见过的多数项目的路由文件。当然，其中也不乏做为取值范围的常量限制，都是值得效仿的。

## Controller Levels
Controller都会被划分层次，一般的是：

```
Application_controller > Base_controller > Version_controller

```

这是一个继承的顺序，按照这个顺序，所属controller中的代码作用域降低，这样就能很好的体现方法的作用域，同时为逻辑层的处理搭好基础，不仅仅是接下来要说的验证会用到，仅仅作为一个例子可以证明这个结构的合理性。
还有，引入代码块，外部类等等，都可以在这里做好限制，避免不必要的混乱。

## Request authentication
在Controller分层之后，我们可以对应在层中写入不同等级的校验方法，例如：Application中写入全局校验的方法，Base中写入API请求的校验方法，以上两种都是具有比较统一的规则来编写校验方法。  
具体如每个方法中需要的参数校验有很多不同方式来实现，个人觉得直接放在Version中去写，清晰明了。

```

      desc 'Create a status.'
      params do
        requires :status, type: String, desc: 'Your status.'
      end
      post do
        authenticate!
        Status.create!({
          user: current_user,
          text: params[:status]
        })
      end


```
这是grape中的一段代码，直接在方法实现的地方说明参数的相关信息，还可以编辑不同错误信息和代号，对错误处理也很友好。单独的gem，rails_params的方式也是类似的实现方式。

## Data process
校验完毕后，就是要组装数据，这就是业务逻辑要解决的范畴，不过记住用好Rails提供的机制，在构建一个优秀API server的时候，是必不可少的。 环境变量，环境配置文件, helper method, lib module, concern, 甚至service 和 middleware都是可以酌情使用，旨在能更好的封装逻辑，包装代码，增加可用性和管理性。

## Data formatting
数据往往需要一个既定的格式送出，所以如何在业务层高效的送出正确的数据的前提下，做到更好的代码管理和维护，就是一个比需要讨论的话题，这个针对不同的业务场景，会有不同的设置方式。  
比如，as_json, jbuilder, serializer 三种方式都可以实现格式化数据，不过其特点各异，所以适用的场景也不同。  

```

as_json: 直接在modle文件中重写as_json方法，这样在数据导出时候就会调用这个方法，客观上达到了包装数据的目的。不过只适合的情景简单的情况下使用，但是最简单。  
jbuilder: 可以通过直接定义文件来规定请求的数据返回格式。
serializer: 可以直接在serializer文件夹中定义要导出的数据格式，实现与jbuilder类似。
grape: grape用了 entity的方式，我觉得定义起来很简单，可以复用，也很明确，确实是个不错的实现方式。


```

## Error handle
处理请求的错误是必不可少的一个环节，错误反馈是一个API server是否优秀的代表。一般的，反馈需要给出一个既定的错误代码以及尽可能准确定位错误的提示信息。代码层，可以封装module，来做错误处理。可以定义异常再返回或者直接返回错误代码的方式来完成，错误信息可以通过配置文本来做处理。  
关于错误处理，目前似乎没有成熟的gem来解决这个问题，不过见到过一个比较好的实践 [link](https://github.com/rails-api/active_model_serializers/issues/983)。  

当然以上所说，还是需要根据项目体积来决定，个人看法是，解决问题时第一位的，最适合的就是最好的。再好的架构，再牛逼的写法，如果不能跑，然并卵。
