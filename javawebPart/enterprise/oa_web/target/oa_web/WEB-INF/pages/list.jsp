<%@ page language="java" contentType="text/html; charset=UTF-8"
         pageEncoding="UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    <link rel="stylesheet" href="https://cdn.bootcss.com/bootstrap/3.3.7/css/bootstrap.min.css"/>
    <script src="https://cdn.bootcss.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
    <title>Insert title here</title>
    <style type="text/css">
        #images{
            width: 50px;
            height: 50px;
        }
    </style>
</head>
<body>
<jsp:include page="top.jsp"/>
<table class="table table-bordered table-hover">
    <tr>
        <th>序号</th>
        <th>图片名称</th>
        <th>图片分类</th>
        <th>模型名称</th>
        <th>数据集名称</th>
    </tr>
    <c:forEach items="${list}" var="picture" >
        <tr>
            <th>${picture.id}</th>
            <th><c:if test="${picture.filename !=null }">
                <img id="images" alt="" src="/images/${picture.filename }">
            </c:if> </th>
            <th>${picture.classes}</th>
            <th>${picture.modelname}</th>
            <th>${picture.datamarket}</th>
        </tr>
    </c:forEach>
</table>

</body>
</html>
