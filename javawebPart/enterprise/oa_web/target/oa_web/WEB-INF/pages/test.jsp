<%@ page language="java" contentType="text/html; charset=UTF-8"
         pageEncoding="UTF-8"%>
<!doctype html>
<%@page import="org.springframework.http.HttpRequest"%>
<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c"%>
<html>
<head>
    <meta charset="utf-8">
    <title>管理员登录</title>
    //导入的jQuery路径
    <script type="text/javascript" src="/theme/js/jquery.min.js"></script>
</head>

<body>
<div class="login">登 录
    <div class="login-form">
        <form action="/AdminLogin" method="post">
            <input type="text" name="adminname" placeholder="用户名" /><br/>
            <input type="password" name="password" placeholder="密码" /><br/>
            <input type="submit" value="登录" /><br/>
        </form>
    </div>

    <div class="user">
        <button onclick="selectAdmin()">查询</button><br/>
    </div>
</div>
</body>

<script>
    //我分开几个函数写主要是好看，但最好写到一起，别人好一起理解
    function selectAdmin() {
        $.ajax({
            //以POST传参，路径为url，不用date传到后台直接查询全部，返回类型json，我只写了成功，失败自己在加点就可以了
            type:"POST",
            url:"/selectAdmin",
            dataType:"json",
            success:function (date) {
                //调用参数名，date就为返回的json数据
                showResult(date);
            }
        })
    }
    function showResult(date) {
        //定义显示数据的位置，根据div的class属性
        var target = $(".login .user ");
        //使按键消失，只按一次
        target.html("");
        //继续定义路径
        var table = "<table border='1'>"
            +"<tr>"
            +"<th>用户</th>"
            +"<th>密码</th>"
            +"</tr>";
        //取出所有匹配的数据，row定义为代表有几条数据
        for (var row=0;row<date.length;row++){
            table =	table
                +"<tr>"
                +"<td>"+date[row].name+"</td>"
                +"<td>"+date[row].password+"</td>"
                +"</tr>";
            +"</table>";
        }
        //显示所有读取数据
        target.append(table);
    }
</script>
</html>
