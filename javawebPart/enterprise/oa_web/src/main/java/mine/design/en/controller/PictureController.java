package mine.design.en.controller;


import mine.design.en.biz.PictureBiz;
import mine.design.en.entity.Picture;
import mine.design.en.global.Contant;
import org.apache.commons.collections.bag.SynchronizedSortedBag;
import org.apache.commons.fileupload.FileItem;
import org.apache.commons.fileupload.FileUploadException;
import org.apache.commons.fileupload.disk.DiskFileItemFactory;
import org.apache.commons.fileupload.servlet.ServletFileUpload;
import org.python.antlr.ParseException;
import org.python.antlr.ast.Str;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.net.Socket;
import java.util.*;

@Controller("pictureController")
public class PictureController {
    @Autowired
    private PictureBiz pictureBiz;
    //对总的所有预测的图片进行展示
    @RequestMapping("/list.do")
    public String list(Model model) {
        List<Picture> list = pictureBiz.getAll();
        model.addAttribute("list", list);
        System.out.println(list);
        return "sum_list";
    }


    //首先是有一个映射到前端的界面让人们去选什么文件，什么模型，什么数据集
    @RequestMapping("/to_add")
    public String toAdd(Map<String,Object> map){
        map.put("picture",new Picture());
        map.put("mlist", Contant.getModels());
        map.put("dlist",Contant.getDataMarket());
        return "picture_add";
    }
    @RequestMapping("/addProduct.do")
    @ResponseBody
    public String addPicture(@RequestParam(value = "file",required = false)MultipartFile file,
                             @RequestParam(value="modelname") String modelname,
                             @RequestParam(value="datamarket") String datamarket,HttpServletRequest request
    ) throws ParseException {
        Picture picture = new Picture();
        String filePath = "D:\\Material";
        String alies = "Material";
        String newFileName = fileOperate(file,filePath);
        String newFilePath=filePath+"\\"+newFileName;
        String[] recieve=getPythonDemo(newFilePath,modelname,datamarket).split("/#/");
        String classes=recieve[0];
        String time_last=recieve[1];
        String accurate=recieve[2];
        picture.setClasses(classes);
        picture.setTime_last(time_last);
        picture.setAccurate(accurate);
        picture.setFilename(newFileName);
        picture.setModelname(modelname);
        picture.setDatamarket(datamarket);
        pictureBiz.add(picture);
        return "redirect:list.do";
    }
    /**
     * 封装操作文件方法， 添加用户 和修改用户都会用到
     * @param file
     * @param filePath
     * @return
     */
    private String fileOperate(MultipartFile file,String filePath) {
        String originalFileName = file.getOriginalFilename();//获取原始图片的扩展名
        System.out.println("图片原始名称："+originalFileName);
        String newFileName = originalFileName;  //新的文件名称
        System.out.println("新的文件名称："+newFileName);
        File targetFile = new File(filePath,newFileName); //创建新文件
        try {
            file.transferTo(targetFile); //把本地文件上传到文件位置 , transferTo()是springmvc封装的方法，用于图片上传时，把内存中图片写入磁盘
        } catch (IllegalStateException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return newFileName;
    }
    //对每个模型预测图片所用时间的展示以看到对比
    @RequestMapping("/time_list.do")
    public String list_time(Model model) {
        List<Picture> list = pictureBiz.getTime();
        model.addAttribute("list", list);
        System.out.println(list);
        return "time_list";
    }

    //对每个模型预测图片所用时间的展示以看到对比
    @RequestMapping("/accurate_list.do")
    public String list_accurate(Model model) {
        List<Picture> list = pictureBiz.getAccurate();
        model.addAttribute("list", list);
        System.out.println(list);
        return "accurate_list";
    }



    public static void sendMsg(Socket socket, String msg) {
        try {
            PrintWriter printWriter = new PrintWriter(socket.getOutputStream(), true);
            printWriter.println(msg);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static String acceptMsg(Socket socket) {
        String line = null;
        try {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                line=bufferedReader.readLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return line;
    }

    public String getPythonDemo(String filePath,String modelname,String datamarket) {
        String data = null;
        System.out.println("start");
        try {
            final Socket socket = new Socket("127.0.0.1", 9529);
            if (socket.isConnected()) {
             // sendMsg(socket, filePath + "/#/E:\\CNN\\Python\\VGG19\\weights2.h5");
                sendMsg(socket, filePath + "/#/"+modelname+"/#/"+datamarket);
                data = acceptMsg(socket);
                System.out.println("返回的数据：" + data);
                try {
                    socket.close();
                    socket.shutdownInput();
                    socket.shutdownOutput();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }
}



